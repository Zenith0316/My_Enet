"""EGIInet 训练、验证和 checkpoint 保存流程。"""

import logging
import os
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from run.test import test_net
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from utils.point_cloud_geometry import PointCloudLocalGeometry
from torch.optim.lr_scheduler import *

def train_net(cfg):
    """按配置训练 EGIInet，并在每个 epoch 结束时进行验证。

    一个训练 batch 的核心数据流是：
    ``(view, gt_pc, part_pc) -> FPS -> model -> CD + 0.01*FT-Loss + w*法向量损失``。
    """

    # 让 cuDNN 为固定尺寸的卷积选择更快算法；可能增加首次运行开销。
    torch.backends.cudnn.benchmark = True

    # 清单中的样本由 ViPCDataLoader 转换为图像、完整点云和残缺点云。
    # view_align=False 会随机选取同一物体的图像/GT 视角。
    ViPC_train = ViPCDataLoader(r'/mnt/d/0916program/elevate/EGllnet/EGIInet/data_pick_list/train_plane_10000_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)
    # drop_last=True 保证每个 batch 尺寸一致；prefetch_factor 控制 worker 预取数量。
    train_data_loader = DataLoader(ViPC_train,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.CONST.NUM_WORKERS,
                              shuffle=True,
                              drop_last=True,
                              prefetch_factor=cfg.CONST.DATA_perfetch)

    # 训练期间使用测试清单作为验证集，类别跟随 TRAIN.CATE。
    ViPC_test = ViPCDataLoader(r'/mnt/d/0916program/elevate/EGllnet/EGIInet/data_pick_list/test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    val_data_loader = DataLoader(ViPC_test,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=cfg.CONST.NUM_WORKERS,
                                   shuffle=True,
                                   drop_last=True,
                                   prefetch_factor=cfg.CONST.DATA_perfetch)

    # 按类别和时间创建 checkpoint、TensorBoard 日志目录。
    output_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TRAIN.CATE, '%s', datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # 分别记录训练 batch/epoch 指标和验证指标。
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    
    # 构建模型并放到 GPU；多 GPU 时使用 DataParallel 包装。
    model = EGIInet()#.apply(weights_init_normal)
    model.cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # 只把 requires_grad=True 的参数交给 Adam。
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # 与编码器计算目标法向量时使用相同的邻域半径，工具会跟随 recon 的设备。
    normal_geometry = PointCloudLocalGeometry(radius=0.1)

    # 先做 GradualWarmup，之后在 LR_DECAY_STEP 指定的 epoch 使用 MultiStepLR 衰减。
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    # EasyDict 中存在 WEIGHTS 键就会恢复模型和优化器状态；从零训练时应删除该键。
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE
        logging.info('Recover complete.')

    print('recon_points: ',cfg.DATASETS.SHAPENET.N_POINTS, 'Parameters: ', sum(p.numel() for p in model.parameters()))
    #exit()
    # 训练主循环：每个 epoch 完整遍历 train_data_loader。
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        model.train()
        # 调整 config_vipc.py 中 NORMAL_LOSS_WEIGHT 后重新启动训练即可生效。
        # 一个 epoch 内保持权重固定，保证 batch 与 epoch 的加权日志含义一致。
        style_weight = 1e-2
        normal_weight = float(cfg.TRAIN.NORMAL_LOSS_WEIGHT)

        total_cd_pc = 0
        total_style = 0
        total_normal = 0
        total_loss = 0

        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        logging.info('Loss weights: CD=1, FT=%g, normal=%g', style_weight, normal_weight)
        with tqdm(train_data_loader) as t:
            # DataLoader 返回顺序固定为：图像、完整 GT、残缺输入。
            for batch_idx, (view,gt_pc,part_pc,) in enumerate(t):

                partial = part_pc.cuda()#[B,N_part,3]
                gt = gt_pc.cuda()#[B,N_gt,3]
                png = view.cuda()

                # 将输入 partial 与 GT 都统一采样为 N_POINTS（当前 2048）。
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)

                
                # recon 是补全点云；style_loss 是编码器的 FT-Loss。
                recon,style_loss,normal_targets=model(partial,png)
                # DataParallel 会将各 GPU 的标量 loss 汇总为向量，先求均值。
                style_loss = style_loss.mean()
                # 每个样本有效点数可能不同，用列表保存真正筛选后的集合。
                # valid_center_points[b] / valid_center_normals[b] 均为 [Kb,3]，
                # 两者逐点对应；Kb 可以为零，损失函数会跳过空集合。
                valid_center_points = [
                    points[mask] for points, mask in zip(normal_targets["points"], normal_targets["valid"])
                ]
                valid_center_normals = [
                    normals[mask] for normals, mask in zip(normal_targets["normals"], normal_targets["valid"])
                ]

                # Chamfer loss 约束补全点云与 GT 的整体空间位置。
                cd = chamfer_sqrt(recon,gt)

                # 使用之前选出的有效中心，在生成点云 recon 中重新估计法向量，
                # 与原始点云的目标法向量比较；梯度通过预测侧传回 recon。
                normal_loss = normal_consistency_loss(
                    recon, valid_center_points, valid_center_normals, geometry=normal_geometry,
                )

                # 三项共同参与 backward；法向量权重在 config_vipc.py 中设置。
                loss_total = (
                    cd
                    + style_weight * style_loss
                    + normal_weight * normal_loss
                )

                # 标准反向传播：清梯度 -> 求梯度 -> 更新参数。
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # 三项统一记录原始真实值，不再混用 CD*1000、FT*10 等倍率。
                # raw 用于观察各自变化，weighted 才是各项对总 loss 的数值贡献。
                cd_pc_item = cd.item()
                total_cd_pc += cd_pc_item
                style_item = style_loss.item()
                total_style += style_item
                normal_item = normal_loss.item()
                total_normal += normal_item
                loss_item = loss_total.item()
                total_loss += loss_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                raw_losses = {'CD': cd_pc_item, 'FT': style_item, 'normal': normal_item}
                weighted_losses = {
                    'CD': cd_pc_item,
                    'FT': style_weight * style_item,
                    'normal': normal_weight * normal_item,
                }
                for name, value in raw_losses.items():
                    train_writer.add_scalar('Loss/Batch/raw/' + name, value, n_itr)
                for name, value in weighted_losses.items():
                    train_writer.add_scalar('Loss/Batch/weighted/' + name, value, n_itr)
                train_writer.add_scalar('Loss/Batch/total', loss_item, n_itr)
                train_writer.add_scalar('Loss/Weight/normal', normal_weight, n_itr)
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                # 科学计数法保留小量级损失，防止四位小数把非零 loss 显示成 0。
                t.set_postfix({
                    'CD': '%.2e' % cd_pc_item,
                    'FT': '%.2e' % style_item,
                    'normal': '%.2e' % normal_item,
                    'FT_w': '%.2e' % weighted_losses['FT'],
                    'normal_w': '%.2e' % weighted_losses['normal'],
                    'lambda_n': '%g' % normal_weight,
                    'total': '%.2e' % loss_item,
                })
                # 训练开头按 batch 推进预热调度器。
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_style = total_style / n_batches
        avg_normal = total_normal / n_batches
        avg_loss = total_loss / n_batches

        # 预热后每个 epoch 推进一次 MultiStepLR。
        lr_scheduler.step()
        for name, raw, weight in (('CD', avg_cdc, 1.0), ('FT', avg_style, style_weight),
                                  ('normal', avg_normal, normal_weight)):
            train_writer.add_scalar('Loss/Epoch/raw/' + name, raw, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/weighted/' + name, weight * raw, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/total', avg_loss, epoch_idx)
        logging.info(
            '[Epoch %d/%d] raw: CD=%.6e FT=%.6e normal=%.6e | '
            'weighted: CD=%.6e FT=%.6e normal=%.6e | lambda_normal=%g total=%.6e',
            epoch_idx, cfg.TRAIN.N_EPOCHS, avg_cdc, avg_style, avg_normal,
            avg_cdc, style_weight * avg_style, normal_weight * avg_normal, normal_weight, avg_loss,
        )

        # 用验证集返回的 CD-L1 选择最佳 checkpoint。
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        # 到达保存周期或验证指标刷新时保存 checkpoint。
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    train_writer.close()
    val_writer.close()

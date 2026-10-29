"""点云采样与距离损失。

当前训练主路径使用 ``farthest_point_sample``、``chamfer_sqrt`` 和
``normal_consistency_loss``；测试使用
``calc_cd`` 计算 CD-L1、CD-L2 与 F1。EMD、单向 CD、DCD 是备用指标。
"""

import torch
import torch.nn
from utils.point_cloud_geometry import PointCloudLocalGeometry
#from models.model_utils import fps_subsample
from metrics.CD.chamfer3D import dist_chamfer_3D
from metrics.CD.fscore import fscore
#from models.Transformer_utils import knn_point,index_points
from metrics.EMD.emd_module import emdModule

# CUDA Chamfer/EMD 算子在模块导入时实例化；本项目默认依赖 GPU 扩展。
chamfer_dist = dist_chamfer_3D.chamfer_3DDist()

emd_dist=emdModule()
import utils.furthestPointSampling.fps as fps


def farthest_point_sample(xyz, npoints):
    """用 FPS 从点云中选取分布较均匀的 ``npoints`` 个点。

    Args:
        xyz: 点云坐标 ``[B,N,3]``。
        npoints: 目标点数。
    Returns:
        采样后的坐标 ``[B,npoints,3]``。
    """
    idx = fps.furthest_point_sample(xyz, npoints)

    # CUDA gather 接受 [B,C,N]，所以采样前后各转置一次。
    new_points = fps.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    return new_points


def emd_loss(x1,x2):
    """计算 Earth Mover's Distance；当前训练主路径未调用。"""
    dis, assigment = emd_dist(x1, x2, 0.05, 3000)
    emd=torch.sqrt(dis).mean()
    return emd


def chamfer(p1, p2):
    """双向平方 Chamfer Distance 的全 batch 均值。"""
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    """计算训练使用的双向 CD-L1。

    CUDA 算子返回平方欧氏距离；这里先开平方再平均两个方向，最终返回标量。
    点集数量可以不同，因此也可比较预测点数与 GT 点数不一致的情况。
    """
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def normal_consistency_loss(recon, center_points, target_normals, geometry=None):
    """在原始有效中心处，比较 recon 与观测点云的局部 PCA 法向量。

    Args:
        recon: 生成点云 Tensor [B,N,3]，保留梯度。
        center_points: 长度为 B 的序列，各项为有效查询坐标 [Kb,3]。
        target_normals: 长度为 B 的序列，各项为对应目标法向量 [Kb,3]。
        geometry: 复用的 PointCloudLocalGeometry；默认使用半径 0.1。
            应与原始目标法向量使用相同的邻域配置。

    返回所有有效对应位置上 mean(1 - abs(dot(n_recon,n_target)))。
    两侧法向量归一化，忽略 PCA 正负号差异。中心坐标与目标法向量 detach，
    只有 recon 侧参与反向传播。空样本、零目标法向量及 recon 的退化邻域
    不参与平均；全部无效时返回可反向传播的零损失。
    此损失不约束没有有效法向量的位置，其空间覆盖仍由 Chamfer loss 约束。
    """
    # 1. 检查生成点云的形状，保证每个批次包含 N 个三维坐标。
    if recon.ndim != 3 or recon.shape[-1] != 3 or min(recon.shape[:2]) < 1:
        raise ValueError("recon must have shape [B,N,3], with B,N > 0")
    # 坐标集合和目标法向量集合必须各有 B 项，逐项对应 recon 中的样本。
    if len(center_points) != recon.shape[0] or len(target_normals) != recon.shape[0]:
        raise ValueError("center_points and target_normals must match the recon batch size")
    # 2. 如果没有传入几何工具，使用与编码器默认设置一致的半径 0.1。
    if geometry is None:
        geometry = PointCloudLocalGeometry(radius=0.1)
    # 3. 初始化损失总和与有效对应点总数。
    # 用 recon.sum()*0 而非普通常数 0，确保全部无效时 loss 仍连接计算图，
    # 可以正常 backward，且对 recon 的梯度为零。
    loss_sum = recon.sum() * 0.0
    valid_count = recon.new_zeros(())
    # 4. 逐样本处理，因为每个样本的有效中心数 Kb 可能不同。
    for b, (centers, normals) in enumerate(zip(center_points, target_normals)):
        # 中心与目标法向量都来自原始残缺点云，是固定监督数据。
        # 放到 recon 所在设备，并通过 detach 切断它们的梯度。
        centers = torch.as_tensor(centers, device=recon.device).detach()
        normals = torch.as_tensor(normals, device=recon.device).detach()
        # 检查一个坐标对应一个法向量，且没有 NaN 或无穷值。
        if centers.ndim != 2 or centers.shape[-1] != 3 or normals.shape != centers.shape:
            raise ValueError("each center/normal pair must have matching shape [Kb,3]")
        if not torch.isfinite(centers).all() or not torch.isfinite(normals).all():
            raise ValueError("centers and target normals must be finite")
        if centers.shape[0] == 0:
            # 该样本没有有效监督中心，不参与损失和有效点数统计。
            continue
        # 5. 使用相同的中心坐标，在生成后的 recon[b] 中寻找邻居并做 PCA。
        # 不再重新选中心，保证预测法向量与目标法向量在同一位置进行比较。
        # differentiable=True 保留预测侧梯度，使损失能更新生成点云。
        predicted = geometry.compute_normals(recon[b], centers, differentiable=True)
        normals = normals.to(dtype=predicted.dtype)
        # 6. 求每个法向量的长度，形状均为 [Kb]。
        # PCA 无法确定方向时返回零向量，两侧长度都非零才参与比较。
        predicted_length = predicted.norm(dim=-1)
        target_length = normals.norm(dim=-1)
        valid = (predicted_length > 0) & (target_length > 0)
        # 7. 将法向量归一化为单位向量，消除长度对点积的影响。
        # unsqueeze(-1): [Kb] -> [Kb,1]，让长度同时除到 x/y/z 分量。
        # 无效项的分母临时设为 1，避免除零；后续通过 valid 排除这些项。
        predicted = predicted / torch.where(valid, predicted_length, torch.ones_like(predicted_length)).unsqueeze(-1)
        normals = normals / torch.where(valid, target_length, torch.ones_like(target_length)).unsqueeze(-1)
        # 8. 逐对点积得到夹角余弦；对点积取绝对值，忽略 PCA 的正负歧义。
        # 这里不是对法向量的三个分量分别取绝对值。
        # clamp 避免浮点误差导致 |dot| 略大于 1，产生负损失。
        dot = (predicted * normals).sum(dim=-1).abs().clamp(max=1.0)
        # 9. 偏差为 1-|dot|：平行/反向平行为 0，垂直为 1。
        # 仅累计有效对应点；按点数加权，而不是让不同 Kb 的样本等权平均。
        loss_sum = loss_sum + (1.0 - dot)[valid].sum()
        valid_count = valid_count + valid.sum()
    # 10. 用整个 batch 的有效点总数求平均；全部无效时分母取 1，返回零。
    return loss_sum / valid_count.clamp_min(1)


def chamfer_single_side(pcd1, pcd2):
    """只计算 pcd1 到 pcd2 的单向平方最近邻距离。"""
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    """只计算 pcd1 到 pcd2 的单向欧氏最近邻距离。"""
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


# def calc_cd(output, gt, calc_f1=False):
#     cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     dist1, dist2, _, _ = cham_loss(gt, output)
#     cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#     cd_t = (dist1.mean(1) + dist2.mean(1))
#     if calc_f1:
#         f1, recall, precision = fscore(dist1, dist2)
#         return cd_p, cd_t, f1
#     else:
#         return cd_p, cd_t

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    """按样本计算测试所需的 Chamfer 指标及可选 F1。

    ``dist1`` 表示 GT 中每个点到 output 的最近平方距离，``dist2`` 表示反向；
    ``cd_p`` 是先开根号的 CD-L1，``cd_t`` 是平方距离形式的 CD-L2。
    参数 ``normalize`` 当前保留但未使用。
    """
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    # cham_loss = cd()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    # separate=True 时不合并两个方向，便于分别分析完整性与准确性。
    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    # F1 使用 metrics/CD/fscore.py 中默认阈值 0.001。
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    """计算 Density-aware Chamfer Distance；当前训练/测试主路径未调用。

    它根据多个点映射到同一最近邻的次数重新加权，惩罚预测点聚集。
    """
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1/idx1：每个 GT 点在预测 x 中的最近距离和索引；dist2/idx2 为反方向。
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res

if __name__ == '__main__':
    # 直接运行时用一维随机点测试 chamfer_sqrt。
    x1 = torch.rand(4, 1024).cuda()
    x2 = torch.rand(4, 1024).cuda()
    x1 = x1.unsqueeze(2)
    x2 = x2.unsqueeze(2)
    print(x1)
    print(x2)
    cd=chamfer_sqrt(x1,x2)
    print(cd)

"""ShapeNetViPC 数据集读取与预处理。

每个样本返回三项数据：物体图像、完整点云（GT）和残缺点云（partial）。
该加载器还负责视角选择、点云坐标对齐以及尺度归一化。
"""

import os
from pathlib import Path
from runtime_paths import project_path
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm


def collate_fn(batch):
    """过滤读取失败产生的 None，再用 PyTorch 默认方式拼成一个 batch。

    当前 train.py 和 test.py 没有把此函数传给 DataLoader，因此它属于备用工具。
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def rotation_z(pts, theta):
    """将形状为 [N, 3] 的点云绕 z 轴旋转 theta 弧度。"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    # 每个点以行向量保存，因此右乘旋转矩阵的转置。
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    """将形状为 [N, 3] 的点云绕 y 轴旋转 theta 弧度。"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    """将形状为 [N, 3] 的点云绕 x 轴旋转 theta 弧度。"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T


class ViPCDataLoader(Dataset):
    """为 PyTorch 提供 ShapeNetViPC 样本的 Dataset。

    Args:
        filepath: 数据清单路径。每行格式为“类别ID/物体ID/视角ID”，
            例如 ``02691156/85fd.../17``。
        data_path: ShapeNetViPC 数据集根目录。
        status: 数据集用途说明（如 train/test），当前只用于打印样本数量。
        pc_input_num: 残缺点云允许的最少点数；不足时重复已有点补齐。
        view_align: 图像是否使用清单中残缺点云对应的同一视角。
        category: 要保留的物体类别英文名；``all`` 表示保留全部类别。

    Returns:
        单个样本为 ``(views, pc, pc_part)``：
        views 是图像张量 [3, H, W]，pc 是完整点云 [N, 3]，
        pc_part 是残缺点云 [M, 3]。
    """

    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        super(ViPCDataLoader,self).__init__()

        # 保存初始化参数；status 和 category 也便于调试时查看数据集实例的用途。
        self.pc_input_num = pc_input_num
        self.status = status

        # filelist 保存清单中的所有行；key 保存经过类别过滤后真正使用的行。
        # cat 当前未被训练/测试代码使用，保留它是为了兼容原项目的数据结构。
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align

        # 将便于阅读的类别名称映射到 ShapeNet taxonomy ID。
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }

        if category != 'all' and category not in self.cat_map:
            raise ValueError('Unknown ShapeNet category: %s' % category)
        if pc_input_num <= 0:
            raise ValueError('pc_input_num must be positive')

        # 兼容 UTF-8/BOM、CRLF、空行和末行无换行；统一清单的路径分隔符。
        filepath = project_path(filepath)
        data_path = project_path(data_path)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, start=1):
                key = line.strip().replace('\\', '/')
                if not key:
                    continue
                parts = key.split('/')
                if (len(parts) != 3 or not parts[0] or not parts[1]
                        or parts[0] in ('.', '..') or parts[1] in ('.', '..')
                        or not parts[2].isascii() or not parts[2].isdigit()
                        or not 0 <= int(parts[2]) < 24):
                    raise ValueError('%s:%d: expected category/object/view (00-23), got %r'
                                     % (filepath, line_num, key))
                self.filelist.append('/'.join((parts[0], parts[1], parts[2].zfill(2))))

        # 数据集的三个子目录分别保存残缺点云、完整点云和渲染图像。
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        for directory in (self.imcomplete_path, self.gt_path, self.rendering_path):
            if not os.path.isdir(directory):
                raise FileNotFoundError('Missing ShapeNetViPC directory: %s; check VIPC_DATA' % directory)

        # 按指定类别过滤清单。路径第一段就是 ShapeNet taxonomy ID。
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)
        if not self.key:
            raise ValueError('No samples for category %r in %s' % (category, filepath))

        # 图像预处理：将较短边缩放到 224，再转换成 [C,H,W]、值域 [0,1] 的张量。
        # ShapeNetViPC 的渲染图通常是正方形，因此处理后通常为 [C,224,224]。
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        # 输出经过类别过滤后的实际样本数。
        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        """读取并预处理索引 idx 对应的一个样本。"""

        # key 形如“类别ID/物体ID/残缺点云视角ID”。
        taxonomy, object_id, part_view_id = self.key[idx].split('/')
        pc_part_path = Path(self.imcomplete_path) / taxonomy / object_id / (part_view_id + '.dat')

        # 图像/GT 随机视角保持原实验行为；不再依赖末尾换行和字符串截断。
        image_view_id = part_view_id if self.view_align else '%02d' % random.randint(0, 23)
        pc_path = Path(self.gt_path) / taxonomy / object_id / (image_view_id + '.dat')
        rendering_dir = Path(self.rendering_path) / taxonomy / object_id / 'rendering'
        view_path = rendering_dir / (image_view_id + '.png')
        with Image.open(view_path) as image:
            views = self.transform(image.convert('RGB'))

        # 从 ShapeNetViPC-GT 读取完整点云，作为网络需要预测的真值。
        with open(pc_path,'rb') as f:
            pc = np.asarray(pickle.load(f, encoding='latin1'), dtype=np.float32)

        # 从 ShapeNetViPC-Partial 读取残缺点云，作为网络的点云输入。
        with open(pc_part_path,'rb') as f:
            pc_part = np.asarray(pickle.load(f, encoding='latin1'), dtype=np.float32)

        # .dat 按原始 pickle 格式读取；latin1 兼容 Python 2 生成的 NumPy pickle。
        for path, points in ((pc_path, pc), (pc_part_path, pc_part)):
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                    or not np.isfinite(points).all()):
                raise ValueError('Expected nonempty finite [N, 3] point cloud: %s' % path)

        # 某些残缺点云少于 pc_input_num 个点时，循环重复已有点并截取到目标数量。
        # 后续 train.py/test.py 还会使用 FPS 将其采样为配置中的 N_POINTS（默认 2048）。
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # 元数据路径按目录拼接，兼容 Windows、Linux 及含点号的目录名。
        view_metadata = np.loadtxt(rendering_dir / 'rendering_metadata.txt', encoding='utf-8-sig', ndmin=2)
        if (view_metadata.shape[0] <= max(int(part_view_id), int(image_view_id))
                or view_metadata.shape[1] < 2 or not np.isfinite(view_metadata[:, :2]).all()):
            raise ValueError('Invalid rendering metadata: %s' % rendering_dir)

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        # 先撤销/转换残缺点云原视角的旋转，再将它旋转到所选图像视角的坐标系。
        # 因此即使 view_align=False，残缺点云和图像仍能在进入网络前保持空间对应。
        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # 使用完整点云的中心和最大半径，同时归一化完整点云与残缺点云：
        # 1. 用完整点云均值平移，使完整点云中心位于原点；
        # 2. 除以完整点云中距离原点最远点的距离，使其最大半径为 1。
        # 两者使用完全相同的参数，从而保持相对位置和尺度一致。
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        if pc_L_max <= 0:
            raise ValueError('Cannot normalize zero-radius point cloud: %s' % pc_path)
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        # DataLoader 会在这些样本维度前增加 batch 维度：
        # views -> [B,3,H,W]，pc -> [B,N,3]，pc_part -> [B,M,3]。
        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        """返回类别过滤后的样本数量，供 DataLoader 计算迭代次数。"""
        return len(self.key)


if __name__ == "__main__":
    # 直接执行本文件时运行一个简单的数据读取示例；被其他模块导入时不会执行。
    from torch.utils.data import DataLoader
    from config_vipc import cfg
    category = cfg.TEST.CATE
    ViPCDataset = ViPCDataLoader(cfg.DATASETS.SHAPENET.TEST_LIST,
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
                                status='test', category=category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    for image, gt, partial in tqdm(train_loader):
        # 示例当前只打印图像 batch 的形状，用于快速确认数据能否正常读取。
        print(image.shape)

        pass

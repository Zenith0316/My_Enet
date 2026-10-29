"""ShapeNetViPC 数据集读取与预处理。

每个样本返回三项数据：物体图像、完整点云（GT）和残缺点云（partial）。
该加载器还负责视角选择、点云坐标对齐以及尺度归一化。
"""

import os
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

        # 逐行读取训练或测试清单。这里保留了行尾换行符，构造路径时再移除。
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()

        # 数据集的三个子目录分别保存残缺点云、完整点云和渲染图像。
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        # 按指定类别过滤清单。路径第一段就是 ShapeNet taxonomy ID。
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

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
        key = self.key[idx]

        # 残缺点云始终使用清单给出的视角。
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')

        # view_align=True：图像、完整点云使用与残缺点云相同的视角编号。
        # view_align=False：为同一物体随机选择 00~23 中的一个图像视角。
        # 当前训练和测试代码都传入 False，因此同一样本每次读取到的图像可能不同。
        if self.view_align:
            ran_key = key        
        else:
            # key 的末尾是“两位视角编号 + 换行符”，先移除这 3 个字符再拼接随机编号。
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')

        # 完整点云和图像使用 ran_key 指定的视角。
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')

        # 兼容原始清单中少数格式异常的行：如果最后一段不只是两位视角编号，
        # 就把最后两位作为视角编号，前面的字符并回物体 ID 后重新构造路径。
        if(len(ran_key.split('/')[-1])>3):
            print("bug")
            print(ran_key.split('/')[-1])
            fin = ran_key.split('/')[-1][-2:]
            interm = ran_key.split('/')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        # 读取渲染图并应用上面定义的缩放、张量转换。
        views = self.transform(Image.open(view_path))

        # 只保留 RGB 三个通道，避免源图带有 alpha 通道时得到 4 通道输入。
        views = views[:3,:,:]

        # 从 ShapeNetViPC-GT 读取完整点云，作为网络需要预测的真值。
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)

        # 从 ShapeNetViPC-Partial 读取残缺点云，作为网络的点云输入。
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)

        # 某些残缺点云少于 pc_input_num 个点时，循环重复已有点并截取到目标数量。
        # 后续 train.py/test.py 还会使用 FPS 将其采样为配置中的 N_POINTS（默认 2048）。
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # 从文件名取得图像和残缺点云各自的两位视角编号。
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]

        # 每个物体的 rendering_metadata.txt 记录 24 个相机视角的信息。
        # 前两列分别用于取得 theta、phi；元数据中的角度值需转换为弧度。
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

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
    category = "table"
    ViPCDataset = ViPCDataLoader('test_list2.txt',data_path='/home/aiello/ShapeNetViPC-Dataset',status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    for image, gt, partial in tqdm(train_loader):
        # 示例当前只打印图像 batch 的形状，用于快速确认数据能否正常读取。
        print(image.shape)

        pass

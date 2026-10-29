"""EGIInet 的图像/点云 token 化、共享编码和显式特征迁移损失。"""

import numpy as np
from torch import einsum
from models.layers.subsample import furthest_point_sample, random_sample
from models.layers.group import KNNGroup, QueryAndGroup, get_aggregation_feautres
from models.layers.conv import  create_convblock2d,create_linearblock
from models.layers.local_aggregation import CHANNEL_MAP
from models.layers.norm import create_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.point_cloud_geometry import PointCloudLocalGeometry
import collections.abc
from itertools import repeat


# 将标量或可迭代对象统一转换为固定长度元组，写法来自 PyTorch 内部工具。
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def gram_matrix(tensor):
    """计算 token 特征的通道 Gram 矩阵。

    输入 ``tensor`` 为 ``[B, N, C]``，N 是 token 数、C 是特征维度；输出为
    ``[B, C, C]``。Gram 矩阵描述通道之间的整体相关性，不要求两种模态的
    token 在空间位置上一一对应。
    """
    gram = einsum('b i n, b n j  -> b i j', tensor.transpose(1,2),tensor)
    return gram


class img_tokenizer(nn.Module):
    """使用无重叠卷积把图像切成 patch token。

    当前配置下：``[B,3,224,224] -> [B,192,16,16] -> [B,256,192]``。
    """

    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768,
                 norm_layer=None, 
                 flatten=True, 
                 bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算出每个维度上 patch 的数量
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 计算总 patch 数量
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # kernel_size 与 stride 都等于 patch_size，因此各 patch 互不重叠；
        # 输出通道 embed_dim 就是每个图像 token 的特征维度。
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """返回 token 序列及卷积后二维网格的高、宽。"""
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            # 把二维 patch 网格展平为 Transformer 需要的 token 序列。
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class pc_tokenizer(nn.Module):
    """通过逐级采样和局部邻域聚合生成点云 token。

    每个 stage 先用 FPS/随机采样选中心点，再用 Ball Query 或 KNN 找邻域，
    最后用共享卷积和池化把一个局部点集压缩成一个中心点 token。
    """

    def __init__(self,
                 sample_ratio=0.0625,
                 scale=4,
                 group_size=32,
                 in_channels=3,
                 layers=4,
                 embed_dim=384,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 normalize_dp=False,
                 radius=0.1,
                 feature_type='dp_df',
                 relative_xyz=True,
                 norm_args={'norm': 'in2d'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 reduction='max',
                 return_group_points=False,
                 **kwargs
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.group_size = group_size
        self.scale=scale
        self.feature_type = feature_type

        # 选择中心点采样算法。主配置使用 FPS，使中心点尽量均匀覆盖点云。
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample

        # 选择中心点邻域构造方式。主配置默认为固定半径 Ball Query。
        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_xyz=relative_xyz, normalize_dp=normalize_dp,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # 根据目标采样率和每层缩减倍数计算 stage 数。
        # 当前 sample_ratio=1/8、scale=2，所以 stages=3：2048->1024->512->256。
        stages = int(math.log(1/sample_ratio, scale))

        # 通道数会在每个 stage 后翻倍；这里反推第一层通道数，使最后一层
        # 达到调用方指定的 embed_dim（当前 pc_h_hidden_dim=192）。
        embed_dim = int(embed_dim // 2 ** (stages-1))
        self.convs = nn.ModuleList()
        self.channel_list = [in_channels]
        for _ in range(int(stages)):
            # 每个 stage 有两段 1x1 Conv2d：先编码邻域点，再融合局部全局特征。
            channels = [CHANNEL_MAP[feature_type](in_channels)] + [embed_dim] * (layers // 2) + [embed_dim * 2] * (
                    layers // 2 - 1) + [embed_dim]
            conv1 = []
            for i in range(layers // 2):
                conv1.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args if i!=(layers//2-1) else None,
                                                act_args=act_args if i!=(layers//2-1) else None,
                                                **conv_args))
            conv1 = nn.Sequential(*conv1)

            channels[layers // 2] *= 2
            conv2 = []
            for i in range(layers // 2, layers):
                conv2.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args,
                                                **conv_args
                                                ))
            conv2 = nn.Sequential(*conv2)
            self.convs.append(nn.ModuleList([conv1, conv2]))

            self.channel_list.append(embed_dim)
            in_channels = embed_dim
            embed_dim *= 2

        # 在邻域点维度上池化。主配置使用最大池化提取最显著的局部特征。
        if reduction in ['mean', 'avg', 'meanpool', 'avgpool']:
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)
        else:
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.out_channels = channels[-1]
        self.re_gr_p=return_group_points


    def forward(self, p, f=None):
        """逐级生成中心点坐标与对应特征。

        Args:
            p: 点坐标 ``[B,N,3]``。
            f: 点特征 ``[B,C,N]``；首次调用传入坐标转置 ``[B,3,N]``。

        Returns:
            out_p: 各 stage 的点坐标列表。
            out_f: 各 stage 的点特征列表。
        """
        B, N, _ = p.shape[:3]
        out_p, out_f = [p], [f]
        
        for convs in self.convs:
            # 逐级下采样：scale=2 表示这一层保留 N/2 个中心点。
            cur_p, cur_f = out_p[-1], out_f[-1]
            idx = self.sample_fn(cur_p, int(N //self.scale)).long()
            N = N // self.scale

            # 按采样索引同时取出中心点坐标和中心点已有特征。
            center_p = torch.gather(cur_p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            center_f = torch.gather(cur_f, 2, idx.unsqueeze(1).expand(-1, cur_f.shape[1], -1))

            # 为每个中心点查询 group_size 个邻居，并构造相对坐标/特征描述。
            dp, fj = self.grouper(center_p, cur_p, cur_f)
            fj = get_aggregation_feautres(center_p, dp, center_f, fj, self.feature_type)

            # 第一段卷积编码每个邻居，再把邻域池化结果广播回每个邻居并拼接，
            # 让第二段卷积同时看到局部逐点信息与邻域整体信息。
            fj = convs[0](fj)
            fj = torch.cat(
                [self.pool(fj).expand(-1, -1, -1, self.group_size),
                fj],
                dim=1)

            # 第二段卷积后沿邻域维池化，得到每个中心点的单个 token 特征。
            out_f.append(self.pool(convs[1](fj)).squeeze(-1))
            out_p.append(center_p)
            
        # 调试模式可额外返回最后一次分组的相对坐标 dp。
        if self.re_gr_p:
            return out_p, out_f,dp
        else:
            return out_p, out_f

class transfer_loss_shared_encoder(nn.Module):
    """统一编码两种模态，并计算 FT-Loss 与跨模态融合结果。

    图像 token 和点云 token 共用同一组 ViT Block。各层中间特征用于计算
    信息迁移/结构保持损失，最后使用一次 MultiheadAttention 更新点云 token。
    """

    def __init__(self,
                 embed_dim=192, 
                 block_head=12, 
                 depth=3,
                 img_patch_size=14,
                 sample_ratio=0.125, 
                 scale=2,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 pc_h_hidden_dim=384,
                 fuse_layer_num=1,
                 **kwargs
                 ):
        super().__init__()
        from timm.models.vision_transformer import Block

        # 两个 tokenizer 最终都产生 [B,256,embed_dim] 形状的 token 序列。
        self.im_to_token=img_tokenizer(patch_size=img_patch_size,embed_dim=embed_dim)
        self.pc_to_token=pc_tokenizer(sample_ratio=sample_ratio,scale=scale,embed_dim=pc_h_hidden_dim)
        # 在原始残缺点云中，用同一球邻域筛选高密度中心并估计法向量。
        self.pc_geometry = PointCloudLocalGeometry(radius=0.1)
        self.embed_dim=embed_dim
        self.fuse_layer_num=fuse_layer_num
        
        # 两种模态逐层调用同一个 blocks[i]，即共享 Transformer 参数。
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=block_head,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(depth)])
        
        # 最终的跨模态注意力。batch_first=True 表示输入形状为 [B,N,C]。
        self.cross_layer=nn.MultiheadAttention(embed_dim, block_head, batch_first=True)
        self.cross_norm=nn.LayerNorm(embed_dim)

        self.depth=depth
        self.pc_norm = create_norm(norm_args, self.embed_dim)

        self.im_norm = create_norm(norm_args, self.embed_dim)
        # 将点云 tokenizer 输出通道对齐到共享 Transformer 的 embed_dim。
        self.proj = nn.Linear(self.pc_to_token.out_channels, self.embed_dim)

        # 用中心点 xyz 坐标生成可学习的位置嵌入，补充点云 token 的空间位置。
        self.pc_pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )

    def forward(self,pc,im):
        """返回融合特征、图像特征、token 中心、FT-Loss 和法向量监督目标。

        第五项 normal_targets 包含 points/normals [B,K,3] 和 valid [B,K]。
        valid=False 的位置补零，计算 loss 时必须排除。
        """
        B=pc.size(0)

        # 点坐标同时作为初始点特征；主配置下最终得到 256 个中心点。
        p_list,x_list=self.pc_to_token(pc,pc.transpose(1,2).contiguous())
        
        # 最后一层提供 M 个候选中心，通常 M=256。密度在原始 pc 中计算，
        # 按密度降序选最多 M//4 个不同坐标（通常 64 个），然后估计法向量。
        final_center_p, final_center_density, final_center_mask = self.pc_geometry.select_dense_centers(
            pc, p_list[-1], keep_ratio=0.25,
        )
        final_center_normals = self.pc_geometry.compute_normals(pc, final_center_p)  # [B,K,3]
        # 高密度不保证可拟合平面：排除补齐位置，以及共线/重合等退化邻域。
        # 后续使用坐标、密度和法向量时，应共同使用 final_center_valid 掩码。
        final_center_valid = final_center_mask & (final_center_normals.norm(dim=-1) > 0)
        final_center_normals = final_center_normals.masked_fill(~final_center_valid.unsqueeze(-1), 0)
        # 保存有效坐标及其对应法向量；无效位置的坐标也清零。
        # 保留固定 K 维并回传掩码，支持 DataParallel 汇总各 GPU 的输出。
        # detach 后作为原始观测点云提供的固定监督目标，不参与反向传播。
        valid_center_points = final_center_p.masked_fill(~final_center_valid.unsqueeze(-1), 0).detach()
        valid_center_normals = final_center_normals.detach()
        normal_targets = {
            "points": valid_center_points,
            "normals": valid_center_normals,
            "valid": final_center_valid,
        }

        # x_list 中点特征采用 [B,C,N]，转置并投影后变为 [B,N,embed_dim]。
        # token 分支仍保留全部 M 个中心，与 x_list[-1] 的特征逐点对齐。
        cent_p,pc_f=p_list[-1],self.proj(x_list[-1].transpose(1, 2))
        im_f,_,_=self.im_to_token(im)

        pc_pos_emd = self.pc_pos_embed(cent_p)

        # 保存每一层的中间特征，稍后跨层计算 FT-Loss。
        pc_f_list=[]
        im_f_list=[]

        for i in range(self.depth):
            # 点云显式加入中心坐标位置嵌入；当前图像分支没有额外的位置嵌入。
            # 两种模态调用同一个 Block 实例，从而映射到共享的特征空间。
            pc_f = self.blocks[i](pc_f + pc_pos_emd)
            im_f = self.blocks[i](im_f)
            pc_f_list.append(pc_f)
            im_f_list.append(im_f)
        pc_f=self.pc_norm(pc_f)
        im_f=self.im_norm(im_f)

        # 注意实参顺序：Query=点云、Key=图像、Value=点云。
        # 图像通过 Q-K 相似度引导权重，而被汇聚的内容仍来自点云。由于 Key 与
        # Value 长度必须相同，当前实现要求图像和点云 token 数均为 256。
        pc_value = pc_f
        if pc_value.shape[1] != im_f.shape[1]:
            pc_value = F.interpolate(
                pc_value.transpose(1, 2), size=im_f.shape[1],
                mode='linear', align_corners=False
            ).transpose(1, 2)
        x, _ = self.cross_layer(pc_f, im_f, pc_value)
        pc_f=self.cross_norm(pc_f+x)

        # fuse_layer_num=0 时只比较两种模态最后一层的 Gram 矩阵。
        if self.fuse_layer_num==0:
            im_target_gram = gram_matrix(im_f_list[-1])
            pc_target_gram = gram_matrix(pc_f_list[-1])
            style_transfer_loss = torch.mean((im_target_gram - pc_target_gram) ** 2) / (self.embed_dim*256)
        else:
            # target 是最后一层，style 是向前回溯 fuse_layer_num 层的特征。
            # 双向 Gram 损失对应信息迁移项 L_infor。
            im_target_gram = gram_matrix(im_f_list[-1])
            im_style_gram = gram_matrix(im_f_list[-(self.fuse_layer_num+1)])
            pc_target_gram = gram_matrix(pc_f_list[-1])
            pc_style_gram = gram_matrix(pc_f_list[-(self.fuse_layer_num+1)])
            im_style_loss = torch.mean((im_target_gram - pc_style_gram) ** 2)
            pc_style_loss = torch.mean((pc_target_gram - im_style_gram) ** 2)
            style_loss = (im_style_loss+pc_style_loss) / (self.embed_dim*256)#info

            # 点云前后层的 MSE 对应结构保持项 L_stc，防止迁移破坏几何表达。
            pc_content_loss=torch.mean((pc_f_list[-1]-pc_f_list[-(self.fuse_layer_num+1)])**2)#str
            style_transfer_loss=pc_content_loss+style_loss

        # pc_f 会进入 decoder；im_f 和 cent_p 当前仅作为辅助输出保留。
        return pc_f,im_f,cent_p,style_transfer_loss,normal_targets
           
if __name__ == '__main__':
    # 直接运行时执行一次编码器 GPU 前向测试，查看形状、损失和参数量。
    import time
    pc = torch.rand([4, 2048, 3]).cuda()
    img = torch.rand([4, 3, 224, 224]).cuda()
    model = transfer_loss_shared_encoder().cuda()
    s = time.time()
    x1,x2, p,s_l,normal_targets= model(pc,img)
    e = time.time()
    print(e-s)
    print(x1.shape)
    print(x2.shape)
    print(s_l)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")


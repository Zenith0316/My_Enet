"""EGIInet 模型总装配。

主数据流：残缺点云和单视图图像 -> 共享跨模态编码器 -> 点云补全解码器。
本文件只负责连接编码器与解码器，具体 token 化、FT-Loss 和点生成逻辑分别位于
``models/encoders.py`` 与 ``models/dec_net.py``。
"""

import numpy as np
#from deformable_attention import DeformableAttention1D
from torch import nn

from models.dec_net import Decoder_Network
import torch
from torch.nn import Module
from models.encoders import  transfer_loss_shared_encoder
from config_vipc import cfg


class EGIInet(Module):
    """图像引导的点云补全网络。

    默认参数直接从 ``config_vipc.py`` 读取，因此通常只需调用 ``EGIInet()``。
    """

    def __init__(self, 
                 embed_dim=cfg.NETWORK.EGIInet.embed_dim,
                 depth=cfg.NETWORK.EGIInet.depth,
                 img_patch_size=cfg.NETWORK.EGIInet.img_patch_size,
                 pc_sample_rate=cfg.NETWORK.EGIInet.pc_sample_rate,
                 pc_sample_scale=cfg.NETWORK.EGIInet.pc_sample_scale,
                 fuse_layer_num=cfg.NETWORK.EGIInet.fuse_layer_num,
                 ):
        super().__init__()

        # 编码器把图像和点云转换为相同维度的 token，经过共享 Transformer、
        # FT-Loss 约束和 cross-attention，输出融合后的点云 token 特征。
        self.encoder=transfer_loss_shared_encoder(embed_dim=embed_dim,
                                               img_patch_size=img_patch_size,
                                               sample_ratio=pc_sample_rate,
                                               scale=pc_sample_scale,
                                               block_head=cfg.NETWORK.shared_encoder.block_head,
                                               depth=depth,
                                               pc_h_hidden_dim=cfg.NETWORK.shared_encoder.pc_h_hidden_dim,
                                               fuse_layer_num=fuse_layer_num,
                                               )

        # 解码器使用融合后的点云 token 生成新点，并拼接从输入残缺点云保留的点。
        self.decoder=Decoder_Network(K1=embed_dim,K2=embed_dim,N=embed_dim)

    def forward(self,pc,img):
        """执行一次点云补全。

        Args:
            pc: 残缺点云，形状通常为 ``[B, 2048, 3]``。
            img: 对应物体的 RGB 图像，形状通常为 ``[B, 3, 224, 224]``。

        Returns:
            final: 补全结果，当前配置下形状为 ``[B, 2048, 3]``。
            style_loss: 编码器计算的 FT-Loss，训练时乘 0.01 加入总损失。
            normal_targets: 原始点云的法向量监督目标字典，包含
                points/normals [B,K,3] 和 bool valid [B,K]。
                同一位置的坐标与法向量一一对应，使用时按 valid 排除补零项。
        """

        # 编码器还返回最终图像 token 和中心点坐标，但当前解码器不直接使用它们。
        feature, _, _, style_loss, normal_targets = self.encoder(pc=pc, im=img)

        # 优先保留具有有效法向量的中心坐标；解码器逐样本用 FPS 补足到 1024 点。
        final = self.decoder(
            feature, pc, center_points=normal_targets["points"], center_mask=normal_targets["valid"],
        )
        return final,style_loss,normal_targets


if __name__ == '__main__':
    # 直接运行本文件时进行一次 GPU 前向测试，并统计可训练参数量。
    import time
    model = EGIInet().cuda()
    pc = torch.rand([4, 2048, 3]).cuda()
    img = torch.rand([4, 3, 224, 224]).cuda()
    s=time.time()
    fine, style_loss, normal_targets = model(pc, img)
    e=time.time()
    print(e-s)
    print(fine.shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")

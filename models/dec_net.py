"""EGIInet 点云补全解码器及若干备用点云模块。

主路径只使用 ``MappingNet``、``AXform`` 和 ``Decoder_Network``。文件前半部分的
KNN、EdgePoolingLayer、PointNet 在当前 EGIInet.forward() 中没有被调用。
"""

import math

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os

import os
import sys
import inspect

# 将项目根目录加入导入路径，便于直接运行本文件进行调试。
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from config_vipc import cfg
from utils.loss_utils import farthest_point_sample


def knn(x, k):
    """计算每个点的 k 个近邻索引（当前模型主路径未使用）。

    输入 x 为 ``[B,C,N]``。代码使用负平方距离，因此 topk 最大值对应最近邻。
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
   # idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """构造 DGCNN 风格的边特征 ``[邻居-中心, 中心]``。"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def batched_index_select(x, dim, index):
    """按照每个 batch 各自的索引，从指定维度收集元素。"""
    for i in range(1, len(x.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


class EdgePoolingLayer(nn.Module):
    """根据动态图边特征评分并保留高分点；当前主模型未使用。"""

    def __init__(self, in_channels, k, ratio=0.5, scoring_fun="tanh", num_points=-1):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.ratio = ratio
        self.score_layer = nn.Conv2d(
            in_channels * 2, 1, kernel_size=1, bias=True)
        self.scoring_fun = scoring_fun
        self.num_points = num_points

    def __str__(self):
        return 'EdgePoolingLayer(in_channels=' + str(self.in_channels) + ', k=' + str(self.k) + ', ratio=' + str(
            self.ratio) + ', scoring_fun=' + str(self.scoring_fun) + ', num_points=' + str(self.num_points) + ')'

    def forward(self, feat, idx=None):
        """输入 ``[B,C,N]``，返回经过评分和 Top-K 选择的点特征。"""
        batch_size, dim, in_points = feat.size()  # (batch_size, α, in_points)
        assert dim == self.in_channels

        # Dynamic Edge Conv
        # Re-computing graph before pooling
        # (batch_size, α * 2, in_points, self.k)
        x = get_graph_feature(feat, k=self.k, idx=idx)
        x = self.score_layer(x)  # (batch_size, 1, in_points, self.k)

        # scores = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1, in_points)
        # ReLU applied to Scores
        # (batch_size, 1, in_points)
        scores = F.relu(x.max(dim=-1, keepdim=False)[0])

        if self.num_points < 0:
            # default - 'num_points' not specified, use 'ratio'
            num_keypoints = math.floor(in_points * self.ratio)
        else:
            # do not use ratio but sample a fixed number of points 'num_points'
            assert self.num_points < in_points, \
                "Pooling more points (%d) than input ones (%d) !" % (
                    self.num_points, in_points)
            num_keypoints = self.num_points

        top, top_idxs = torch.topk(scores.squeeze(), k=num_keypoints, dim=1)
        new_feat = batched_index_select(feat.permute(
            0, 2, 1), 1, top_idxs)  # (batch, num_keypoints, α)
        top = top.unsqueeze(2)

        # Apply scoring function!
        if self.scoring_fun == 'tanh':
            new_feat = new_feat * torch.tanh(top)
        elif self.scoring_fun == 'softmax':
            new_feat = new_feat * F.softmax(top, dim=1)
        elif self.scoring_fun == 'leaky-relu':
            new_feat = new_feat * F.leaky_relu(top, negative_slope=0.001)

        return new_feat

class PointNet(nn.Module):
    """简化 PointNet 全局特征提取器；当前主模型未使用。"""

    def __init__(self):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x

class MappingNet(nn.Module):
    """把共享编码特征映射为一个解码分支的隐式点特征。

    主配置输入为 ``[B,256,192]``。线性层作用于最后一维，BatchNorm 则要求
    通道位于中间维，因此代码在每层之间反复 permute。
    """

    def __init__(self, K1,N=256):
        super(MappingNet, self).__init__()
        self.K1 = K1
        self.N=N

        self.fc1 = nn.Linear(self.N, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.K1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        # [B,256,192] --fc1--> [B,256,512] --转置--> [B,512,256]
        x = F.relu(self.bn1(self.fc1(x).permute(0,2,1)))
       
        x = x.permute(0,2,1)
        x = F.relu(self.bn2(self.fc2(x).permute(0,2,1)))
      
        x = x.permute(0,2,1)
        x = F.relu(self.bn3(self.fc3(x).permute(0,2,1)))
        x = x.permute(0,2,1)
        x = F.relu(self.bn4(self.fc4(x).permute(0,2,1)))
        
        x = x.permute(0,2,1)
       
        return x

class AXform(nn.Module):
    """学习 token 汇聚权重，并将聚合特征映射为三维点坐标。

    ``conv3`` 固定输出 128 组权重，所以每个分支生成 128 个点，返回
    ``[B,128,3]``；构造参数 N 并不直接决定这里的输出点数。
    """

    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.N = N  
        self.conv1 = nn.Conv1d(K1, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=2)
        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        # 保留 MappingNet 输出，后面作为被加权汇聚的 value。
        x_base = x
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
       
        # 生成 [B,128,256] 权重，并沿输入 token 维（dim=2）归一化。
        x_weights = self.softmax(x)
        
        # [B,128,256] @ [B,256,192] -> [B,128,192]。
        x = torch.bmm(x_weights, x_base)  
        
        # 转成 [B,192,128] 后用 1D 卷积将每个聚合特征映射为 xyz。
        x = x.transpose(1, 2).contiguous()
        x = self.conv4(x)
        x = x.transpose(1, 2).contiguous()

        return x

class Decoder_Network(nn.Module):
    """多分支补全解码器。

    8 个独立分支各生成 128 个新点，共 1024 点；然后拼接从输入残缺点云
    保留的 1024 点（优先保留有效中心，再用 FPS 补足），因此最终输出
    ``[B,2048,3]``。
    """

    def __init__(self, K1, K2, N):
        super(Decoder_Network, self).__init__()
        
        self.num_branch = 8
        self.K1 = K1
        self.K2 = K2
        self.N = N
        
        # 每个分支拥有独立参数，让不同分支学习物体的不同局部区域。
        self.featmap = nn.ModuleList([MappingNet(self.K1,self.N) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])
       
    @torch.no_grad()
    def _sample_partial(self, x_part, center_points=None, center_mask=None, num_points=1024):
        """优先保留有效中心，再从剩余观测点中 FPS 补足 num_points 个点。

        center_points: [B,K,3]，来自编码器的筛选中心。
        center_mask: bool [B,K]，仅 True 位置作为初始保留点。
        返回 [B,num_points,3]，每个样本前 x 个点为有效中心，随后为
        FPS 选出的 num_points-x 个点。不同样本的 x 可以不同。
        未传中心时沿用原有 FPS；传中心时排除坐标完全相同的重复候选。
        所有不同坐标不足 num_points 个时，遍历已有点循环补齐。
        """
        if center_points is None:
            if center_mask is not None:
                raise ValueError("center_mask requires center_points")
            return farthest_point_sample(x_part.contiguous(), num_points)
        if (center_points.ndim != 3 or center_points.shape[0] != x_part.shape[0]
                or center_points.shape[-1] != 3):
            raise ValueError("center_points must have shape [B,K,3]")
        if (center_mask is None or center_mask.dtype != torch.bool
                or center_mask.shape != center_points.shape[:2]):
            raise ValueError("center_mask must be bool with shape [B,K]")
        if x_part.ndim != 3 or x_part.shape[-1] != 3 or min(x_part.shape[:2]) < 1:
            raise ValueError("x_part must have shape [B,N,3], with B,N > 0")
        center_points = center_points.to(device=x_part.device, dtype=x_part.dtype)
        center_mask = center_mask.to(device=x_part.device)
        if not torch.isfinite(x_part).all() or not torch.isfinite(center_points[center_mask]).all():
            raise ValueError("partial points and valid centers must be finite")

        batches = []
        for b in range(x_part.shape[0]):
            # 1. 掩码排除补齐中心；按原有顺序保留有效坐标，原点也可以有效。
            centers = center_points[b][center_mask[b]]
            # 上游已经去重，这里也防止直接调用解码器时传入重复中心。
            same = (centers[:, None, :] == centers[None, :, :]).all(dim=-1)
            centers = centers[~same.tril(diagonal=-1).any(dim=-1)]
            num_centers = centers.shape[0]
            if num_centers > num_points:
                raise ValueError("valid centers exceed the partial point budget")

            # 2. 对原始观测点去重，并排除已经保留的中心坐标，避免重复选中。
            candidates = torch.unique(x_part[b], dim=0)
            is_center = (candidates[:, None, :] == centers[None, :, :]).all(dim=-1).any(dim=-1)
            candidates = candidates[~is_center]
            remaining = num_points - num_centers

            # 3. 从剩余候选中 FPS 选出 1024-x 个点；x 是本样本有效中心数。
            if remaining == 0:
                extra = candidates[:0]
            elif candidates.shape[0] > remaining:
                extra = farthest_point_sample(candidates.unsqueeze(0).contiguous(), remaining)[0]
            else:
                # 候选不多于所需数量时全部保留，无需运行 FPS。
                extra = candidates
            partial = torch.cat((centers, extra), dim=0)

            # 4. 极端退化点云的不同坐标可能不足 1024 个，重复已有点补齐。
            # x_part 非空，因此 partial 至少有一个坐标；不会除以零。
            if partial.shape[0] < num_points:
                repeat_indices = torch.arange(num_points - partial.shape[0], device=x_part.device) % partial.shape[0]
                partial = torch.cat((partial, partial[repeat_indices]), dim=0)
            batches.append(partial)
        return torch.stack(batches, dim=0)

    def forward(self, x, x_part, center_points=None, center_mask=None):
        """根据融合 token、残缺点云及可选的有效中心坐标生成补全点云。"""

        # 有效中心优先作为 x_partial，再 FPS 补足到 [B,1024,3]。
        x_partial = self._sample_partial(x_part, center_points, center_mask, num_points=1024)
        x_feat = x
        # 建立 [B,0,3] 空张量，逐分支沿点数维拼接新生成的点。
        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)
        
        x_branch = []
        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat)) 
            x_1 = torch.cat((x_1, _x_1), dim=1) 
            x_branch.append(_x_1)

        # 新生成点与原始观测点合并，得到最终补全结果。
        x_coarse = torch.cat((x_1, x_partial), dim=1)

        return x_coarse

if __name__ == '__main__':
    # 直接运行时的解码器示例。这里的调用缺少构造参数，属于原项目遗留代码。
    x_part = torch.randn(16, 2048, 3).cuda()
    x = torch.randn(16, 128, 256).cuda()
    model = Decoder_Network().cuda()
    out = model(x, x_part)

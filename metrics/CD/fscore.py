"""根据双向最近邻距离计算点云重建 F-score。"""

import torch


def fscore(dist1, dist2, threshold=0.001):
    """计算每个 batch 样本的 F-score。

    dist1、dist2 为 Chamfer CUDA 算子返回的两个方向的平方最近邻距离，形状
    分别为 ``[B,N]``、``[B,M]``。threshold 也处于平方距离尺度。
    """

    # 两个方向分别统计落在阈值内的点占比，再计算调和平均。
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)

    # 两个比例都为 0 时分母为 0，将产生的 NaN 视为 F-score=0。
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


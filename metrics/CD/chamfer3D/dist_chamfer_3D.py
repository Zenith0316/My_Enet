"""Chamfer Distance CUDA 扩展的 JIT 编译与 PyTorch autograd 包装。"""

from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
# chamfer_found = importlib.find_loader("chamfer_3D") is not None
# if not chamfer_found:
## Cool trick from https://github.com/chrdiller
# 首次导入时根据本目录中的 C++/CUDA 源码编译扩展，之后会使用 PyTorch 缓存。
print("Jitting Chamfer 3D")

from torch.utils.cpp_extension import load
p=os.path.abspath(__file__).split('/')[:-1]

"""
chamfer_3D = load(name="chamfer_3D",
        sources=[
            r"D:\0StudyRelated\SVDFormer-main\SVDFormer-main\metrics\CD\chamfer3D\chamfer_cuda.cpp",
            r"D:\0StudyRelated\SVDFormer-main\SVDFormer-main\metrics\CD\chamfer3D\chamfer3D.cu",
            ])
"""

chamfer_3D = load(name="chamfer_3D",
        sources=[
            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
            ])
print("Loaded JIT 3D CUDA chamfer distance")

# else:
#     import chamfer_3D
#     print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    """连接 Python autograd 与 CUDA Chamfer 前向/反向函数。"""

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        # xyz1、xyz2 分别为 [B,N,3]、[B,M,3]，允许 N 与 M 不同。
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        # dist1/dist2 是双向最近邻的平方距离，idx1/idx2 是最近邻索引。
        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    """供损失函数调用的 nn.Module 接口。"""

    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)

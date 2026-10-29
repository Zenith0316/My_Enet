# EGIInet 在 Windows + WSL2 Ubuntu 22.04 中复现

本文面向已经安装好 WSL2、Ubuntu 22.04 和 Conda 的环境。项目使用 CUDA 自定义算子进行点云采样、Chamfer Distance 和 EMD 计算，因此必须使用 NVIDIA GPU；当前代码不能使用 CPU 模式运行。

建议先完成预训练模型测试，确认环境和数据正确后再开始训练。以下命令都在 WSL2 的 Ubuntu 终端中执行，不是在 Windows PowerShell 中执行。

## 1. 进入项目目录

Windows 路径 D:\0916program\elevate\EGllnet\EGIInet 在 WSL 中通常对应：

    cd /mnt/d/0916program/elevate/EGllnet/EGIInet
    pwd
    export EGIINET_ROOT="$(pwd)"

如果项目已复制到 WSL 的 Linux 文件系统，则进入实际路径，例如 cd ~/EGIInet。当前目录应能看到 main.py、config_vipc.py、models、metrics 和 utils。

## 2. 检查 WSL2 GPU 和 CUDA

先在 Windows 主机安装支持 WSL2 的 NVIDIA 显卡驱动，然后在 Ubuntu 中运行：

    nvidia-smi
    which nvcc
    nvcc --version

nvidia-smi 应能显示 GPU；nvcc 应能找到 CUDA 编译器。项目按 CUDA 11.8 验证。WSL 中不要安装 Linux 显卡驱动，Windows 主机负责提供显卡驱动；WSL 内只需要 CUDA Toolkit。

## 3. 创建 Conda 环境

    conda create -n egiinet python=3.9 -y
    conda activate egiinet
    python --version
    python -m pip install --upgrade pip setuptools wheel

安装项目验证过的 PyTorch 版本：

    python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

安装 Python 依赖：

    python -m pip install easydict opencv-python transforms3d h5py timm open3d tensorboardX ninja einops tqdm scikit-learn pillow fast-pytorch-kmeans
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

如果 torch-scatter 只是因为网络或 wheel 暂时失败，可以先跳过；主训练路径不会导入 models/layers/kmeans.py。

## 4. 验证 PyTorch

    python -c "import torch; print('torch =', torch.__version__); print('torch cuda =', torch.version.cuda); print('cuda available =', torch.cuda.is_available()); print('device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

应满足：PyTorch 为 2.1.2，torch cuda 为 11.8，cuda available 为 True，并能打印出 NVIDIA GPU 名称。

## 5. 检查数据集

数据集根目录必须直接包含：

    ShapeNetViPC-Dataset/
    ├── ShapeNetViPC-Partial/
    ├── ShapeNetViPC-GT/
    └── ShapeNetViPC-View/

例如数据集位于 Windows 的 D:\dataset\ShapeNetViPC-Dataset，WSL 路径通常是：

    export VIPC_DATA="/mnt/d/dataset/ShapeNetViPC-Dataset"
    ls -ld "$VIPC_DATA"/ShapeNetViPC-Partial "$VIPC_DATA"/ShapeNetViPC-GT "$VIPC_DATA"/ShapeNetViPC-View

列表中的一行类似 02691156/object_id/00，应对应：

    ShapeNetViPC-Partial/02691156/object_id/00.dat
    ShapeNetViPC-GT/02691156/object_id/00.dat
    ShapeNetViPC-View/02691156/object_id/rendering/00.png
    ShapeNetViPC-View/02691156/object_id/rendering/rendering_metadata.txt

用测试列表的第一条样本检查路径：

~~~bash
python - "$VIPC_DATA" <<'PY'
import os
import sys
data_root = sys.argv[1]
with open('data_pick_list/test_list.txt', 'r', encoding='utf-8') as f:
    key = f.readline().strip()
category, object_id, view_id = key.split('/')
paths = [
    os.path.join(data_root, 'ShapeNetViPC-Partial', category, object_id, view_id + '.dat'),
    os.path.join(data_root, 'ShapeNetViPC-GT', category, object_id, view_id + '.dat'),
    os.path.join(data_root, 'ShapeNetViPC-View', category, object_id, 'rendering', view_id + '.png'),
    os.path.join(data_root, 'ShapeNetViPC-View', category, object_id, 'rendering', 'rendering_metadata.txt'),
]
print('sample key:', key)
for path in paths:
    print(('OK   ' if os.path.isfile(path) else 'MISS '), path)
if not all(os.path.isfile(path) for path in paths):
    raise SystemExit('数据集目录结构或数据集根路径不正确')
PY
~~~

如果列表文件来自 Windows，建议清理 CRLF，避免路径末尾出现隐藏的回车符：

    sed -i 's/\r$//' data_pick_list/train_list.txt data_pick_list/test_list.txt

数据集放在 /mnt/c、/mnt/d 等 Windows 挂载盘上可以运行，但大量小文件读取通常会比 WSL Linux 文件系统慢。

## 6. 修改配置

    nano config_vipc.py

至少修改以下三项：

    __C.DATASETS.SHAPENET.VIPC_PATH = '/mnt/d/dataset/ShapeNetViPC-Dataset'
    __C.DIR.OUT_PATH = '/mnt/d/egiinet_outputs'
    __C.CONST.DEVICE = '0'

默认 TRAIN.CATE 和 TEST.CATE 都是 plane，第一次建议保持不变。第一次只验证流程时，建议临时使用：

    __C.CONST.NUM_WORKERS = 2
    __C.CONST.DATA_perfetch = 2
    __C.TRAIN.BATCH_SIZE = 4
    __C.TEST.BATCH_SIZE = 4

作者配置是训练 batch size 128、测试 batch size 64、8 个 worker；显存不足时必须降低 batch size。

## 7. 修正列表文件的硬编码路径

当前 run/train.py 和 run/test.py 把列表文件写死为：

    /project/EGIInet/data_pick_list/train_list.txt
    /project/EGIInet/data_pick_list/test_list.txt

如果你的仓库不在 /project/EGIInet，分别打开两个文件，将它们改成项目实际的 Linux 路径，例如：

    /mnt/d/0916program/elevate/EGllnet/EGIInet/data_pick_list/train_list.txt
    /mnt/d/0916program/elevate/EGllnet/EGIInet/data_pick_list/test_list.txt

修改后检查：

    rg -n "project/EGIInet|train_list|test_list" run/train.py run/test.py

不要把 Windows 的 D:\... 直接写进 Python 字符串；运行时使用 /mnt/d/...。

## 8. 编译 CUDA 扩展

    sudo apt update
    sudo apt install -y build-essential git libgl1
    conda activate egiinet
    cd "$EGIINET_ROOT"

按顺序执行：

    cd models/pointnet2_batch
    python -m pip install -v .
    cd ../../metrics/CD/chamfer3D
    python -m pip install -v .
    cd ../../EMD
    python -m pip install -v .
    cd ../../utils/furthestPointSampling
    python -m pip install -v .
    cd "$EGIINET_ROOT"

Chamfer Distance 和 EMD 在 Python 导入时还会用 torch.utils.cpp_extension.load JIT 编译。建议设置缓存目录：

    export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_egiinet"
    mkdir -p "$TORCH_EXTENSIONS_DIR"

如果是 RTX 30 系列，可在编译前设置：

    export TORCH_CUDA_ARCH_LIST="8.6"

其他显卡不要照抄 8.6；不确定时先不设置该变量。

## 9. 导入检查

    cd "$EGIINET_ROOT"
~~~bash
python - <<'PY'
import torch
import pointnet2_batch_cuda
import fps_cuda
from metrics.CD.chamfer3D import dist_chamfer_3D
from metrics.EMD.emd_module import emdModule
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('pointnet2: OK')
print('fps: OK')
print('chamfer: OK')
print('emd: OK')
PY
~~~

第一次导入可能会再次编译 Chamfer 和 EMD。最后打印四个 OK 且 cuda 为 True 后再继续。

## 10. 用预训练模型测试

从项目 README 中的 Google Drive 链接下载作者 checkpoint，放到 WSL 可访问的位置，例如：

    /mnt/d/egiinet_checkpoints/plane-ckpt-best.pth

在 config_vipc.py 中设置：

    __C.CONST.WEIGHTS = '/mnt/d/egiinet_checkpoints/plane-ckpt-best.pth'
    __C.TEST.CATE = 'plane'

执行：

    cd "$EGIINET_ROOT"
    conda activate egiinet
    CUDA_VISIBLE_DEVICES=0 python main.py --test

成功时最后应出现 TEST RESULTS、Overall 和 Epoch -1。数据加载器设置为 view_align=False，会随机从 24 个相机视角中选择图像，所以多次测试结果可能有轻微变化。

## 11. 从头训练

确认预训练测试正常后，把权重配置注释掉，避免误加载 checkpoint：

    # __C.CONST.WEIGHTS = '/mnt/d/egiinet_checkpoints/plane-ckpt-best.pth'

先做 1 个 epoch 的冒烟测试：

    __C.TRAIN.N_EPOCHS = 1
    __C.TRAIN.BATCH_SIZE = 4
    __C.TEST.BATCH_SIZE = 4
    __C.CONST.NUM_WORKERS = 2
    __C.CONST.DATA_perfetch = 2

运行：

    cd "$EGIINET_ROOT"
    conda activate egiinet
    CUDA_VISIBLE_DEVICES=0 python main.py

成功标志是能进入 epoch 1，完成训练和验证，并在 OUT_PATH/plane/ 下生成 checkpoints 和 logs。冒烟测试成功后，再恢复作者配置：

    __C.TRAIN.N_EPOCHS = 160
    __C.TRAIN.BATCH_SIZE = 128
    __C.TEST.BATCH_SIZE = 64
    __C.CONST.NUM_WORKERS = 8
    __C.CONST.DATA_perfetch = 8

然后执行：

    CUDA_VISIBLE_DEVICES=0 python main.py

## 12. TensorBoard

训练时另开一个 WSL 终端：

    conda activate egiinet
    tensorboard --logdir /mnt/d/egiinet_outputs/plane/logs --bind_all

在 Windows 浏览器打开 http://localhost:6006。

## 13. 常见问题

### torch.cuda.is_available() 为 False

检查 nvidia-smi、PyTorch 是否安装了 cu118 wheel，以及 Windows NVIDIA 驱动是否支持 WSL2。nvcc 能运行并不等于 PyTorch 已经能访问 GPU。

### CUDA_HOME is not set 或找不到 nvcc

说明 WSL 中没有可用的 CUDA Toolkit，或 PATH 没有包含 CUDA。检查：

    command -v nvcc
    echo "$CUDA_HOME"

### No module named pointnet2_batch_cuda 或 No module named fps_cuda

确认 conda activate egiinet 后，在同一个环境中重新执行第 8 节的 pip install -v .。

### 找不到 .dat、.png 或 rendering_metadata.txt

检查 VIPC_PATH 是否指向包含三个 ShapeNetViPC-* 子目录的数据集根目录，而不是某一个子目录；同时检查 WSL 路径和列表内容。

### FileNotFoundError: /project/EGIInet/data_pick_list/train_list.txt

说明第 7 节的硬编码列表路径尚未修正。

### CUDA out of memory

把 TRAIN.BATCH_SIZE 和 TEST.BATCH_SIZE 降到 1、2 或 4，并将 NUM_WORKERS 降到 0 或 2。确认单步运行后再逐步增加 batch size。

### 编译时报 ninja、undefined symbol 或 ABI 错误

确认 which python 指向 .../envs/egiinet/bin/python，并在同一个环境中编译和运行。如果切换过 PyTorch 版本，建议重建干净 Conda 环境。

### 训练很慢或 DataLoader 卡住

先使用 NUM_WORKERS = 0 验证功能，再改成 2、4、8。位于 /mnt/d 的数据集通过 Windows 文件系统读取大量小文件时可能明显变慢。

## 14. 推荐执行顺序

1. nvidia-smi 和 nvcc --version
2. 创建并激活 egiinet 环境
3. 安装 PyTorch 2.1.2 + CUDA 11.8 和 Python 依赖
4. 验证 torch.cuda.is_available() 为 True
5. 检查数据集三个目录和第一条样本
6. 修改 config_vipc.py
7. 修正 run/train.py、run/test.py 的列表路径
8. 编译 CUDA 扩展并完成导入检查
9. 使用预训练权重执行 python main.py --test
10. 用 1 epoch、小 batch 做训练冒烟测试
11. 恢复正式配置开始完整训练

仓库中的 data_pick_list/train_list.txt 和 data_pick_list/test_list.txt 包含多类别样本，但默认配置只筛选 plane。第一次复现建议保持 plane 不变，完成后再修改 TRAIN.CATE 和 TEST.CATE。

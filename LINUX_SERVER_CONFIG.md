# Linux 服务器配置与硬编码清单

本文记录迁移后的配置入口、目标文件用途以及保留的实验常量。行号对应本次修改后的文件，后续修改代码时可能变化。表格中的路径默认以项目根目录为基准；显式传给分析脚本的命令行路径以当前工作目录为基准。

## 1. 服务器运行

将仓库上传或克隆到服务器后，在仓库目录执行以下命令。`/实际数据目录` 和权重路径需要换成服务器真实位置。`EGIINET_ROOT` 仅用于下面的 shell 命令，Python 自动根据文件位置定位项目。

```bash
export EGIINET_ROOT="$(pwd)"
export PYTHONPATH="$EGIINET_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VIPC_DATA="/实际数据目录/ShapeNetViPC-Dataset"
export EGIINET_OUTPUT="$EGIINET_ROOT/output/2048"
export EGIINET_NUM_WORKERS=2
# 普通单机可自行指定；Slurm 作业保留调度器提供的 CUDA_VISIBLE_DEVICES。
# export CUDA_VISIBLE_DEVICES=0

# 从零训练（默认仍使用 plane 的 10000 条训练清单）
unset EGIINET_WEIGHTS
python -u main.py

# 独立测试：检查点中需要 model 键。
export EGIINET_WEIGHTS="/实际权重目录/ckpt-best.pth"
python -u main.py --test

# 加载模型和优化器训练：检查点还需要 optimizer 键。
# 当前恢复逻辑沿用原实现，不恢复 epoch 和完整调度器状态。
python -u main.py
```

数据集根目录必须包含 `ShapeNetViPC-Partial/`、`ShapeNetViPC-GT/`、`ShapeNetViPC-View/`。清单中的每行 `类别ID/物体ID/视角ID` 用来定位 `.dat` 点云、渲染图和渲染元数据。数据集、权重和训练输出已被 Git 忽略，需要单独传到服务器。

需要完整训练清单时设置：

```bash
export EGIINET_TRAIN_LIST="$EGIINET_ROOT/data_pick_list/train_list.txt"
export EGIINET_TEST_LIST="$EGIINET_ROOT/data_pick_list/test_list.txt"
```

官方权重按 `<category>-ckpt-best.pth` 命名；与当前结构不一致的检查点仍可能加载失败。

```bash
export EGIINET_WEIGHTS_DIR="/实际官方权重目录"
python show_analyze/run_official_inference.py --category plane --count 12 \
  --output "$EGIINET_ROOT/result/inference"
python show_analyze/evaluate_prediction.py --input-root "$EGIINET_ROOT/result/inference"
# 分析脚本按八类、每类 1000 个样本检查完整性；小批量结果会标记不完整。
python show_analyze/analyze_1000_results.py --root "/实际评估结果目录"
```

## 2. 环境与编译

沿用原项目的 Python 3.9 / PyTorch 2.1.2 / CUDA 11.8 环境方案，安装步骤见 `REPRODUCE_WSL2.md`。需要 NVIDIA GPU、与 PyTorch 配套的 CUDA Toolkit、C++ 编译器和 Ninja；本次未改成 CPU 训练。`.cuda()` 和逻辑 `cuda:0` 使用服务器可见 GPU，并不固定物理显卡编号。

在已激活的环境中安装 FPS 和 PointNet2 扩展：

```bash
python -m pip install ./utils/furthestPointSampling
python -m pip install ./models/pointnet2_batch
# CD/EMD 在导入时会从各自源码目录 JIT 编译；也保留独立构建脚本。
bash metrics/CD/chamfer3D/run_build.sh
bash metrics/EMD/run_build.sh
```

构建脚本默认使用当前环境的 `python`，可用 `PYTHON=/实际环境/bin/python` 覆盖。旧集群分区、节点和 `srun` 参数已经移除；使用 Slurm 时在脚本外按实际集群申请资源。`TORCH_EXTENSIONS_DIR` 可指定 PyTorch JIT 缓存位置，`CUDA_HOME` 由服务器 CUDA 安装决定。Shell 文件已通过 `.gitattributes` 固定为 LF。

中文绘图默认查找已安装的中文字体；可设置 `EGIINET_FONT=/实际字体目录/font.ttc`。没有中文字体时指标计算仍可使用，中文图形可能缺字。

## 3. 已修改的环境硬编码

下表列出配置入口及代码消费位置；“原值”指迁移前机器绑定值。

| 文件与当前行号 | 原硬编码 | 当前入口、目标及作用 |
|---|---|---|
| `runtime_paths.py:6` | 无统一根目录 | 按本文件位置确定项目根目录，避免依赖启动目录。 |
| `runtime_paths.py:9` | 散落的机器绝对路径 | 展开 ~ 和环境变量；相对路径按项目根目录解析。 |
| `config_vipc.py:26` | `/home/zenith/datasets/ShapeNetViPC-Dataset` | `VIPC_DATA`；默认 `datasets/ShapeNetViPC-Dataset/`，存放 partial、GT 和视图。 |
| `config_vipc.py:27` | `/mnt/d/0916program/.../train_plane_10000_list.txt` | `EGIINET_TRAIN_LIST`；默认 `data_pick_list/train_plane_10000_list.txt`，训练样本索引，保持原来的 10000 样本实验。 |
| `config_vipc.py:28` | `/mnt/d/0916program/.../test_list.txt` | `EGIINET_TEST_LIST`；默认 `data_pick_list/test_list.txt`，验证与独立测试索引。 |
| `config_vipc.py:51` | `/mnt/d/0916program/.../output/2048` | `EGIINET_OUTPUT`；默认 `output/<N_POINTS>`，保存检查点和 TensorBoard 日志。 |
| `config_vipc.py:52` | `/mnt/d/Dataset/ShapeNetViPC/ShapeNetViPC/checkpoints_new` | `EGIINET_WEIGHTS_DIR`；默认 `checkpoints/`，供官方权重推理脚本按类别找文件。 |
| `config_vipc.py:169` | 注释中的 WSL 检查点路径 | `EGIINET_WEIGHTS`；默认 None；指定单个 .pth 文件，测试加载 model，训练加载 model 和 optimizer。 |
| `config_vipc.py:57` | 固定 `0` 并覆盖系统变量 | 优先 CUDA_VISIBLE_DEVICES，其次 EGIINET_DEVICE，最后 0；保留调度器分配。 |
| `config_vipc.py:58` | main.py 固定 1 | `EGIINET_SEED`，默认 1；控制主入口随机种子。 |
| `config_vipc.py:37` | 固定 2 | `EGIINET_NUM_WORKERS`，默认 2；控制加载进程数，支持 0。 |
| `run/train.py:34` | WSL 训练清单绝对路径 | 读取统一训练清单配置，清单本身应上传 Git。 |
| `run/train.py:46` | WSL 测试清单绝对路径 | 训练期间验证集。 |
| `run/test.py:30` | WSL 测试清单绝对路径 | 独立测试集。 |
| `main.py:11, 12` | 覆盖 CUDA_VISIBLE_DEVICES | 在导入 torch 和 CUDA 扩展前配置 GPU，保留已有服务器环境变量。 |
| `show_analyze/run_official_inference.py:112` | WSL 官方模型目录 | --weights-dir 覆盖目录，--weights 覆盖单文件；--test-list/--output 也可覆盖默认目标。 |
| `show_analyze/analyze_1000_results.py:121` | 固定本机 evaluation_1000_samples_per_category_emd1024 目录 | 改为必须传 --root；读取类别目录中的 metrics.json 并汇总。 |
| `show_analyze/evaluate_prediction.py:33` | WSL Windows 字体路径和固定 Linux 字体路径 | 可选字体文件路径；未设置则用已安装字体名称匹配。 |
| `localize_report_figures.py:18` | 固定本机评估目录 | 必须传 --results；读取 metrics.json 中引用的点云文件，重新生成报告图片。搬运旧结果时 JSON 内部绝对路径也需有效。 |
| `utils/ViPCdataloader.py:239` | test_list2.txt 和 /home/aiello/ShapeNetViPC-Dataset | 数据读取示例改为统一配置；以 python -m utils.ViPCdataloader 从项目根目录运行。 |
| `metrics/EMD/emd_module.py:35, 36` | 注释中的 D:\0StudyRelated\SVDFormer-main 路径及手动拆分 / | 定位同目录 emd.cpp、emd_cuda.cu，用于 JIT 编译 EMD。 |
| `metrics/CD/chamfer3D/dist_chamfer_3D.py:19, 20` | 注释中的 Windows 路径及手动拆分 / | 定位同目录 chamfer_cuda.cpp、chamfer3D.cu，用于 JIT 编译 CD。 |
| `metrics/EMD/clean.sh:5` | 删除旧用户 /mnt/lustre/.../site-packages/emd-*.egg | 仅清理本脚本目录中的缓存和构建文件，不删除其他环境安装内容。 |
| `metrics/EMD/run_build.sh:5` | 固定 dsta/ips_share 分区、SG-IDC/SH-IDC 节点及 GPU 资源参数 | 当前环境 Python 执行；前两个安装扩展，后者 JIT 编译并运行 EMD 示例，需 GPU。 |
| `metrics/CD/chamfer3D/run_build.sh:5` | 固定 dsta/ips_share 分区、SG-IDC/SH-IDC 节点及 GPU 资源参数 | 当前环境 Python 执行；前两个安装扩展，后者 JIT 编译并运行 EMD 示例，需 GPU。 |
| `metrics/EMD/run_compile.sh:5` | 固定 dsta/ips_share 分区、SG-IDC/SH-IDC 节点及 GPU 资源参数 | 当前环境 Python 执行；前两个安装扩展，后者 JIT 编译并运行 EMD 示例，需 GPU。 |

## 4. 配置文件全部显式设置

以下列出所有叶子配置项；数值/类别是保留的实验默认值。无目标文件的项用于模型或训练行为。路径项的实际用途见上一表。

| 文件与行号 | 配置项 | 当前表达式 | 说明 |
|---|---|---|---|
| `config_vipc.py:21` | `cfg.DATASETS.SHAPENET.N_POINTS` | `2048` | partial 和 GT 的 FPS 点数；默认 2048。 |
| `config_vipc.py:26` | `cfg.DATASETS.SHAPENET.VIPC_PATH` | `env_path('VIPC_DATA', 'datasets/ShapeNetViPC-Dataset')` | 数据集目录。 |
| `config_vipc.py:27` | `cfg.DATASETS.SHAPENET.TRAIN_LIST` | `env_path('EGIINET_TRAIN_LIST', 'data_pick_list/train_plane_10000_list.txt')` | 训练索引文件。 |
| `config_vipc.py:28` | `cfg.DATASETS.SHAPENET.TEST_LIST` | `env_path('EGIINET_TEST_LIST', 'data_pick_list/test_list.txt')` | 测试/验证索引文件。 |
| `config_vipc.py:37` | `cfg.CONST.NUM_WORKERS` | `int(os.environ.get('EGIINET_NUM_WORKERS', '2'))` | 数据加载并行进程数。 |
| `config_vipc.py:41` | `cfg.CONST.DATA_perfetch` | `2` | 每个 worker 预取批次数。 |
| `config_vipc.py:51` | `cfg.DIR.OUT_PATH` | `env_path('EGIINET_OUTPUT', 'output/%s' % __C.DATASETS.SHAPENET.N_POINTS)` | 检查点和日志输出目录。 |
| `config_vipc.py:52` | `cfg.DIR.WEIGHTS` | `env_path('EGIINET_WEIGHTS_DIR', 'checkpoints')` | 检查点文件或按类别存放的目录，见第 3 节。 |
| `config_vipc.py:57` | `cfg.CONST.DEVICE` | `os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('EGIINET_DEVICE', '0'))` | GPU 可见性，见第 3 节。 |
| `config_vipc.py:58` | `cfg.CONST.SEED` | `int(os.environ.get('EGIINET_SEED', '1'))` | 主入口随机种子。 |
| `config_vipc.py:68` | `cfg.NETWORK.EGIInet.embed_dim` | `192` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:71` | `cfg.NETWORK.EGIInet.depth` | `6` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:75` | `cfg.NETWORK.EGIInet.img_patch_size` | `14` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:79` | `cfg.NETWORK.EGIInet.pc_sample_rate` | `0.125` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:84` | `cfg.NETWORK.EGIInet.pc_sample_scale` | `2` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:89` | `cfg.NETWORK.EGIInet.fuse_layer_num` | `2` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:95` | `cfg.NETWORK.shared_encoder.block_head` | `12` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:99` | `cfg.NETWORK.shared_encoder.pc_h_hidden_dim` | `192` | 网络结构参数；需与检查点结构匹配。 |
| `config_vipc.py:107` | `cfg.TRAIN.BATCH_SIZE` | `2` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:110` | `cfg.TRAIN.N_EPOCHS` | `100` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:113` | `cfg.TRAIN.SAVE_FREQ` | `25` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:116` | `cfg.TRAIN.LEARNING_RATE` | `0.001` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:119` | `cfg.TRAIN.LR_MILESTONES` | `[16,32,48,64,80,96,112,128,144]` | 预留字段；当前训练未读取。 |
| `config_vipc.py:123` | `cfg.TRAIN.LR_DECAY_STEP` | `[10,20,30,40,50,60,70,80,90,100]` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:127` | `cfg.TRAIN.WARMUP_STEPS` | `1` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:130` | `cfg.TRAIN.GAMMA` | `0.7` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:133` | `cfg.TRAIN.BETAS` | `(.9, .999)` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:136` | `cfg.TRAIN.WEIGHT_DECAY` | `0` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:141` | `cfg.TRAIN.NORMAL_LOSS_WEIGHT` | `0.01` | 法向量一致性损失权重。 |
| `config_vipc.py:146` | `cfg.TRAIN.CATE` | `'plane'` | 类别过滤；all 表示混合类别。 |
| `config_vipc.py:149` | `cfg.TRAIN.d_size` | `1` | 预留字段；当前训练未读取。 |
| `config_vipc.py:158` | `cfg.TEST.METRIC_NAME` | `'ChamferDistance'` | 预留名称；当前实际测试固定计算 CD-L1、CD-L2、F1。 |
| `config_vipc.py:161` | `cfg.TEST.CATE` | `'plane'` | 类别过滤；all 表示混合类别。 |
| `config_vipc.py:164` | `cfg.TEST.BATCH_SIZE` | `1` | 训练/测试超参数；迁移保留原值。 |
| `config_vipc.py:169` | `cfg.CONST.WEIGHTS` | `env_path('EGIINET_WEIGHTS', os.environ['EGIINET_WEIGHTS']) if os.environ.get('EGIINET_WEIGHTS') else None` | 检查点文件或按类别存放的目录，见第 3 节。 |

## 5. 代码中保留的主要实验约定

这些不是服务器文件路径。为避免改变训练结果，保留原值；修改点数、结构或损失时需同时检查相关位置。下表覆盖模型/数据/评估的重要固定值，不逐项枚举张量索引、颜色和打印格式。

| 文件与当前行号 | 固定值或约定 | 作用及修改注意事项 |
|---|---|---|
| `run/train.py:112` | FT 权重 1e-2 | 总损失中 FT-Loss 的倍率，与 NORMAL_LOSS_WEIGHT 分开。 |
| `run/train.py:81` | radius=0.1 | 预测点云侧法向量邻域，需与编码器目标几何定义一致。 |
| `models/encoders.py:254` | radius=0.1 | 目标法向量估计邻域。 |
| `models/encoders.py:103` | radius=0.1 | 点云 tokenizer 局部聚合半径。 |
| `utils/loss_utils.py:89` | radius=0.1 | 未显式传 geometry 时使用的损失几何默认值。 |
| `utils/ViPCdataloader.py:78` | 3500 | 不足此点数的 partial 重复补齐；不是最终 FPS 点数。 |
| `utils/ViPCdataloader.py:133` | 224 | 图像尺寸，需匹配图像 tokenizer。 |
| `utils/ViPCdataloader.py:94` | 8 类 taxonomy 映射 | plane 等类别名指向 ShapeNet 类别目录 ID。 |
| `models/encoders.py:51` | 224 | 图像 tokenizer 默认尺寸。 |
| `models/dec_net.py:262, 328` | 1024 | 解码器保留的 partial 点数；最终输出结构固定，不随 N_POINTS 自动变化。 |
| `show_analyze/run_official_inference.py:100, 101` | 0 | 推理抽样独立使用的随机种子，NumPy/PyTorch 同样固定 0。 |
| `show_analyze/run_official_inference.py:106` | cuda:0 | 首张可见逻辑 GPU；物理 GPU 用 CUDA_VISIBLE_DEVICES 选择。 |
| `show_analyze/analyze_1000_results.py:138` | 1000 | 统计脚本按每类 1000 个结果判断完整性。 |
| `show_analyze/evaluate_prediction.py:240` | [0.001,0.002,0.003,0.005,0.007,0.01] | F-score 距离阈值，可用 --thresholds 覆盖。 |
| `show_analyze/evaluate_prediction.py:241` | 256 | 近似 EMD 采样点数，可用 --emd-points 覆盖。 |
| `show_analyze/plot_prediction.py:62` | 0.01 | 差异点绘图阈值，可用同名参数覆盖。 |
| `run/test.py:78, 79` | 1000 | 展示 CD 指标时乘 1000；F1 不缩放。 |

## 6. 文件命名约定和旧文档

`run/train.py` 写入 `<OUT_PATH>/<CATE>/checkpoints/<时间>/ckpt-best.pth` 或 `ckpt-epoch-NNN.pth`，日志写入同一类别下的 `logs/<时间>/{train,test}`。推理目录包含 `partial.ply`、`gt.ply`、`prediction.ply`、`view.png`、`inference_metrics.txt`；评估/绘图脚本依赖这些名称，以及 `metrics.json` 和 `evaluation_complete.flag`。这些名称保持一致即可迁移。

`命令行/` 下两份命令已改用 `$EGIINET_ROOT`、`$EGIINET_WEIGHTS_DIR`；示例中的时间戳检查点仍需替换成实际训练结果。`REPRODUCE_WSL2.md` 保留旧 WSL 安装记录，其中机器路径仅作为历史示例，Linux 服务器以本文为准。


## 7. 本地验证范围

已检查全部待上传 Python 文件语法；验证默认配置、环境变量覆盖、从其他工作目录启动时的路径、GPU 环境变量优先级，以及 NUM_WORKERS=0/2 时三个 DataLoader 的预取参数。配置验证使用最小 EasyDict 替身，因为当前 Windows Python 环境缺少 easydict。未在目标 Linux/CUDA 环境执行完整训练、CUDA 扩展编译或真实数据读取；迁移后仍需按实际数据、权重和 CUDA 环境完成运行验证。

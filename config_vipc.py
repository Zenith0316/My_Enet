from easydict import EasyDict as edict

# 本文件集中保存数据集、模型、训练和测试参数。
# EasyDict 允许使用 cfg.TRAIN.BATCH_SIZE 这样的“属性访问”方式读取字典配置，
# 效果类似普通字典 cfg['TRAIN']['BATCH_SIZE']，但写法更直观。
__C                                              = edict()
# cfg 是提供给 main.py、run/train.py、run/test.py 等模块使用的公开配置对象。
cfg                                              = __C

# =============================================================================
# 数据集配置
# =============================================================================
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()

# 每个样本最终送入网络的点数。
# 训练和测试时，partial（残缺点云）与 gt（完整点云）都会通过最远点采样 FPS
# 统一采样为该数量；当前为 2048 个点。
__C.DATASETS.SHAPENET.N_POINTS                   = 2048

# ShapeNetViPC 数据集的根目录。
# 该目录下应包含 ShapeNetViPC-Partial、ShapeNetViPC-GT、ShapeNetViPC-View。
# 运行项目前需要根据实际机器修改此绝对路径。
__C.DATASETS.SHAPENET.VIPC_PATH        = '/home/zenith/datasets/ShapeNetViPC-Dataset'

# =============================================================================
# 通用运行参数
# =============================================================================
__C.CONST                                        = edict()

# PyTorch DataLoader 用于并行读取数据的子进程数。
# 数值越大不一定越快，还会增加 CPU 和内存开销。若要设为 0，当前项目还需
# 同时修改 train.py/test.py，使 DataLoader 不再接收 prefetch_factor 参数。
__C.CONST.NUM_WORKERS                            = 2

# DataLoader 中每个 worker 提前加载的 batch 数，即 prefetch_factor。
# 当前配置表示每个数据加载进程最多预取 2 个 batch。
__C.CONST.DATA_perfetch                          = 2

# =============================================================================
# 输出目录与计算设备
# =============================================================================
__C.DIR                                          = edict()

# 训练结果的根目录。
# run/train.py 会在其下按以下结构创建检查点和 TensorBoard 日志：
# OUT_PATH/<训练类别>/checkpoints/<时间>/ 和 OUT_PATH/<训练类别>/logs/<时间>/。
__C.DIR.OUT_PATH                                 = '/mnt/d/0916program/elevate/EGllnet/EGIInet/output/2048'

# 指定程序可见的 GPU 编号，对应 main.py 中的 CUDA_VISIBLE_DEVICES。
# '0' 表示只使用第 0 张 GPU；多卡可写成 '0,1'。
__C.CONST.DEVICE                                 = '0'

# =============================================================================
# 网络结构配置
# =============================================================================
__C.NETWORK                                      = edict()
__C.NETWORK.EGIInet                              = edict()

# 图像 token、点云 token以及解码器使用的主特征维度。
# 当前为 192；该值必须能被注意力头数 block_head 整除。
__C.NETWORK.EGIInet.embed_dim                    = 192

# 共享 Transformer 编码块的层数。层数增加通常会提高计算量和显存占用。
__C.NETWORK.EGIInet.depth                        = 6

# 将 224x224 图像切分为 patch 时，每个正方形 patch 的边长。
# 设为 14 时会得到 16x16，共 256 个图像 token。
__C.NETWORK.EGIInet.img_patch_size               = 14

# 点云 tokenizer 最终保留的采样比例。
# 当前 0.125 = 1/8；输入 2048 个点时，最终产生约 256 个中心点 token。
__C.NETWORK.EGIInet.pc_sample_rate               = 0.125


# 点云每一层的下采样倍数。当前每层将点数缩小为原来的 1/2。
# 它和 pc_sample_rate 一起决定点云 tokenizer 的层数：log_scale(1/sample_rate)。
__C.NETWORK.EGIInet.pc_sample_scale              = 2

# 计算跨模态风格迁移损失时，向前回溯的编码层数。
# 当前为 2，即使用最后一层与倒数第 3 层特征计算风格/内容约束；设为 0 时
# 只比较图像与点云最后一层的 Gram 矩阵。该值应小于 depth。
__C.NETWORK.EGIInet.fuse_layer_num               = 2

__C.NETWORK.shared_encoder                       = edict()

# Transformer 自注意力和图像/点云交叉注意力的 head 数。
# embed_dim=192、block_head=12 时，每个 head 的维度为 16。
__C.NETWORK.shared_encoder.block_head            = 12

# 点云分层 tokenizer 的目标输出通道数，随后会投影到 embed_dim。
# 提高该值会增加点云局部特征提取阶段的参数量和计算量。
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192

# =============================================================================
# 训练配置
# =============================================================================
__C.TRAIN                                        = edict()

# 每次参数更新使用的样本数。增大可能提高吞吐量，但会占用更多显存。
__C.TRAIN.BATCH_SIZE                             = 2

# 总训练轮数；一轮（epoch）表示完整遍历一次训练 DataLoader。
__C.TRAIN.N_EPOCHS                               = 100

# 周期检查点的保存间隔（epoch）。此外，只要验证集 CD 变好也会保存最佳模型。
__C.TRAIN.SAVE_FREQ                              = 25

# Adam 优化器的初始学习率。
__C.TRAIN.LEARNING_RATE                          = 0.001

# 预留的学习率里程碑配置。当前项目代码没有读取该字段，修改它不会生效。
__C.TRAIN.LR_MILESTONES                          = [16,32,48,64,80,96,112,128,144]

# 实际传给 MultiStepLR 的 epoch 节点；到达每个节点时学习率乘以 GAMMA。
# __C.TRAIN.LR_DECAY_STEP                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.LR_DECAY_STEP                          = [10,20,30,40,50,60,70,80,90,100]

# 学习率预热步数，由 GradualWarmupScheduler 使用。
# 当前训练代码在开头按 batch 调度预热，之后按 epoch 调度学习率。
__C.TRAIN.WARMUP_STEPS                           = 1

# MultiStepLR 的学习率衰减系数；例如 0.001 会在节点处变为 0.0007。
__C.TRAIN.GAMMA                                  = 0.7

# Adam 优化器的一阶、二阶动量衰减系数。
__C.TRAIN.BETAS                                  = (.9, .999)

# Adam 的 L2 权重衰减系数；0 表示不使用权重衰减。
__C.TRAIN.WEIGHT_DECAY                           = 0

# 法向量一致性损失权重：总损失 = CD + 0.01*FT + 此权重*normal。
# 设为 0 可关闭该项的加权贡献；修改后重新启动训练生效。
# 训练日志同时显示 raw 与 weighted，调权重时比较 weighted 项和验证指标。
__C.TRAIN.NORMAL_LOSS_WEIGHT                     = 0.01

# 训练和训练期间验证所使用的物体类别。
# 可用名称由 utils/ViPCdataloader.py 的 cat_map 定义，如 plane、car、chair、table；
# 使用 'all' 表示不过滤类别。
__C.TRAIN.CATE                                   = 'plane'

# 预留字段。当前项目没有读取该参数，修改它不会影响训练。
__C.TRAIN.d_size                                 = 1

# =============================================================================
# 独立测试配置
# =============================================================================
__C.TEST                                         = edict()

# 预期的主评估指标名称，但当前 test.py 没有读取该字段。
# 实际测试会固定计算 CD-L1、CD-L2 和 F1，并返回 CD-L1 作为模型选择依据。
__C.TEST.METRIC_NAME                             = 'ChamferDistance'

# 独立运行测试时使用的物体类别；含义和 TRAIN.CATE 相同。
__C.TEST.CATE                                    = 'plane'

# 独立测试时每个 batch 的样本数。
__C.TEST.BATCH_SIZE                              = 1

# 预训练/检查点文件路径。
# 使用 --test 或 --inference 时必须配置；文件中应包含名为 'model' 的状态字典。
# 注意：run/train.py 通过判断 WEIGHTS 这个键是否存在来决定是否恢复训练。因此：
#   - 继续训练：保留此配置，并指向有效检查点（还需包含 'optimizer'）。
#   - 从零训练：需要注释或删除整行，不能只把值改为 None。
# 当前是 Linux/WSL 风格绝对路径，请确认它在实际运行环境中存在。
#__C.CONST.WEIGHTS = '/mnt/d/Dataset/ShapeNetViPC/ShapeNetViPC/checkpoints_new/plane-ckpt-best.pth'

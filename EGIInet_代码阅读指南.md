# EGIInet 代码阅读指南

这份指南把论文 **Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion**（EGIInet）中的方法，映射到当前仓库的真实代码。目标不是逐文件解释，而是告诉你：先读什么、每一部分解决什么问题、输入输出是什么、哪些地方最值得调试。

> `result/` 是实验输出目录，本指南不把它作为模型阅读入口。

## 1. 先建立整体心智模型

任务是 View-guided Point Cloud Completion（ViPC）：

```text
部分点云 P (B, 2048, 3) + 单视图图像 I (B, 3, 224, 224)
        |
        v
数据读取、视角对齐、归一化
        |
        v
统一编码器：图像 token + 点云 token -> 共享 ViT 特征
        |
        v
显式信息交互：Gram/结构迁移损失约束两种模态
        |
        v
一次 cross-attention：点云作 Query/Value，图像作 Key
        |
        v
解码器：8 个分支生成新点 + FPS 保留部分点云
        |
        v
完成点云，与 GT 计算 Chamfer Distance
```

论文的核心不是“把图像直接变成 3D 点”，而是：先让两种模态进入相近的 token 表示，再用显式结构约束使图像关注对补全有用的区域，最后让点云特征主导 3D 解码。

## 2. 推荐阅读顺序

按下面顺序读，最容易把论文概念和代码对应起来：

1. `main.py`：程序入口，区分训练、测试和推理。
2. `config_vipc.py`：所有重要超参数、数据路径、类别和 checkpoint。
3. `utils/ViPCdataloader.py`：确认一个 batch 的三个对象分别是什么，以及视角如何对齐。
4. `models/EGIInet.py`：只看模型总装配和 `forward` 数据流。
5. `models/encoders.py`：重点读 `img_tokenizer`、`pc_tokenizer`、`transfer_loss_shared_encoder`。
6. `models/dec_net.py`：读 8 分支点生成器和部分点云拼接逻辑。
7. `utils/loss_utils.py`：读 FPS、训练用 CD、测试用 CD/F-score。
8. `run/train.py`、`run/test.py`：最后看优化、验证、日志和 checkpoint。
9. `models/layers/`：只有在需要追踪局部邻域、Ball Query、CUDA FPS 或底层张量时再深入。

## 3. 程序入口与配置

### `main.py`

关注 `main()`：

- 无参数：调用 `train_net(cfg)`。
- `--test` 或 `--inference`：要求 `cfg.CONST.WEIGHTS` 非空，然后调用 `test_net(cfg)`。
- `set_seed()` 固定 NumPy 和 PyTorch 随机种子，但数据加载器在 `view_align=False` 时仍会随机选择图像视角。

阅读时先记住：真正的模型不是在 `main.py` 中定义的，`main.py` 只负责把配置交给训练/测试流程。

### `config_vipc.py`

最值得看的配置：

| 配置 | 当前含义 |
|---|---|
| `N_POINTS=2048` | 训练前对输入 partial 和 GT 做 FPS，统一到 2048 点 |
| `embed_dim=192` | 图像/点云 token 的共同通道数 |
| `depth=6` | 共享 Transformer block 数；这是当前配置值 |
| `img_patch_size=14` | 224 图像被划成 16x16=256 个 token |
| `pc_sample_rate=0.125`、`pc_sample_scale=2` | 点云 tokenizer 逐级 FPS，2048 -> 1024 -> 512 -> 256 |
| `block_head=12` | 多头注意力头数 |
| `fuse_layer_num=2` | 最后 2 个共享 block 可视为逻辑 SFT 阶段，并决定 FT-Loss 比较的层间边界 |
| `TRAIN.CATE` / `TEST.CATE` | 默认只筛选 `plane` 类别 |
| `TRAIN.BATCH_SIZE` / `TEST.BATCH_SIZE` | 当前工作区已改为 4，原论文训练规模不要直接假定为当前配置 |
| `CONST.WEIGHTS` | 测试时加载的 checkpoint；训练时若保留该字段也会尝试恢复 |

## 4. 数据流：先弄清楚输入到底是什么

文件：[`utils/ViPCdataloader.py`](utils/ViPCdataloader.py)

### 4.1 三个返回值

`__getitem__()` 返回：

```text
views  : (3, 224, 224) 图像
pc     : (N_gt, 3)    完整 GT 点云
pc_part: (N_part, 3)  部分输入点云
```

训练循环中的解包顺序是 `(view, gt_pc, part_pc)`，因此不要把第二、第三个张量看反。

### 4.2 视角处理是本项目的关键细节

- `view_align=True`：图像视角与列表中 partial 的视角相同。
- `view_align=False`：从同一物体的 24 个视角中随机选一个图像/GT 视角；当前训练和测试都使用 `False`。
- `rendering_metadata.txt` 提供 `theta`、`phi`，代码先把 partial 从原视角旋转到图像坐标系，再用 GT 的均值和最大半径归一化。

这部分对应论文中“单视图可以来自不同观察方向”的设置。调试时优先检查：`key`、`ran_key`、图像编号、metadata 行号和旋转后的点云方向。

### 4.3 数据集目录

`VIPC_PATH` 必须直接包含：

```text
ShapeNetViPC-Partial/
ShapeNetViPC-GT/
ShapeNetViPC-View/
```

`data_pick_list/train_list.txt` 和 `data_pick_list/test_list.txt` 中的 key 形如 `02691156/object_id/00`。类别名会通过 `cat_map` 映射为 ShapeNet synset id。

## 5. 模型总装配：`models/EGIInet.py`

核心代码只有两步：

```python
feature, _, _, style_loss = self.encoder(pc=pc, im=img)
final = self.decoder(feature, pc)
return final, style_loss
```

含义：

- 编码器只把 **点云侧特征** `pc_f` 交给解码器；图像特征通过 cross-attention 的 Key 影响注意力权重，最终仍更新点云特征。
- `style_loss` 实际上是论文所说的 FT-Loss 的代码实现结果，训练时作为辅助损失加入。
- `cent_p`、最终图像特征在当前 `EGIInet.forward()` 中没有继续传给 decoder。

## 6. 统一编码器：`models/encoders.py`

这是最重要的文件，建议逐行读。

### 6.1 图像 token：`img_tokenizer`

```python
self.proj = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
```

224x224 图像经过 14x14、步长 14 的卷积，得到 16x16 个 patch，每个 patch 映射到 192 维：

```text
(B, 3, 224, 224) -> (B, 256, 192)
```

这对应论文的 image tokenizer：每个 token 表示一个局部图像区域，token 顺序保留二维网格的全局结构。

### 6.2 点云 token：`pc_tokenizer`

点云 tokenizer 的关键操作：

1. FPS 选中心点。
2. Ball Query 找每个中心点的局部邻域，默认 `group_size=32`。
3. 用相对坐标和点特征构造局部几何描述。
4. 2D 卷积块聚合邻域特征。
5. 重复 3 个 stage，点数大致为 2048 -> 1024 -> 512 -> 256。
6. 用中心点坐标经过 `pc_pos_embed` 生成位置嵌入。

论文强调“每个 token 表示局部几何，token 序列表示整体结构”。这里的 `center_p` 和 `pc_pos_embed` 正是这句话的代码落点。

### 6.3 共享特征提取：`transfer_loss_shared_encoder`

关键模块：

- `self.blocks`：共享给图像 token 和点云 token 的 ViT `Block`。
- `self.pc_norm`、`self.im_norm`：分别规范化两种模态。
- `self.cross_layer = nn.MultiheadAttention(...)`：最终一次跨模态融合。
- `self.proj`：将点云 tokenizer 的输出通道投影到 `embed_dim=192`。

循环部分：

```python
pc_f = self.blocks[i](pc_f + pc_pos_emd)
im_f = self.blocks[i](im_f)
```

这对应论文的 Unified Encoder / Shared Feature Extractor（SFE）：两种模态使用同一组 Transformer block，从而减小 latent space 的模态差异。

随后：

```python
x, _ = self.cross_layer(pc_f, im_f, pc_f)
pc_f = self.cross_norm(pc_f + x)
```

这里必须按实参顺序读：`Query=pc_f`、`Key=im_f`、`Value=pc_f`。图像特征参与 Query-Key 相似度并决定注意力权重，被加权汇聚的 Value 仍是点云特征；结果回到点云 token 序列。因此图像不是直接提供 3D 坐标，而是引导网络如何重组点云表示。这个调用依赖图像和点云 token 数都为 256，否则 Key 与 Value 的序列长度不一致。

### 6.4 论文 SFTnet 与当前代码的对应关系

论文把“共享特征提取”和“共享特征转移网络（SFTnet）”概念上分开；当前代码没有名为 `SFTnet` 的独立类，而是把两个阶段压进同一个 `self.blocks`。按 `fuse_layer_num` 的层间边界理解：前面的 block 是逻辑 SFE，最后 `fuse_layer_num` 个 block 是逻辑 SFT 阶段。实际实现是：

- `self.blocks` 依次生成各层的点云/图像中间特征 `pc_f_list`、`im_f_list`；
- 用这些中间特征计算 FT-Loss；
- 再用一个 `cross_layer` 完成最终融合。

因此阅读当前仓库时，不要机械寻找 `SFTnet` 类名；应把“最后若干共享 block + 跨越该边界的 FT-Loss”作为 SFTnet 的代码落点，再把 `cross_layer` 看成其后的特征融合。

## 7. 显式信息交互与 FT-Loss

论文的 FT-Loss：

```text
L_transfer = L_infor + L_stc
```

### 7.1 Gram 矩阵：`gram_matrix`

```python
gram = einsum('b i n, b n j -> b i j', tensor.transpose(1,2), tensor)
```

对形状 `(B, N, C)` 的 token 特征，得到 `(B, C, C)` 的通道相关性矩阵。它不要求图像 token 与点云 token 逐位置对应，而是比较整体结构/通道关系。

### 7.2 信息损失 `L_infor`

代码中：

```python
im_style_loss = mean((G(im_target) - G(pc_style)) ** 2)
pc_style_loss = mean((G(pc_target) - G(im_style)) ** 2)
style_loss = (im_style_loss + pc_style_loss) / (embed_dim * 256)
```

它是双向的：

- 图像最终特征的结构，约束点云较早层特征；
- 点云最终特征的结构，约束图像较早层特征。

论文的意图是让图像感知“缺失关系”，让点云感知图像中与缺失部分相关的结构。

### 7.3 结构损失 `L_stc`

代码：

```python
pc_content_loss = mean((pc_f_list[-1] - pc_f_list[-(fuse_layer_num+1)]) ** 2)
```

它只约束点云特征，防止跨模态迁移破坏点云自身的结构表达。

### 7.4 总训练损失

`run/train.py` 中：

```python
cd = chamfer_sqrt(recon, gt)
loss_total = cd + style_loss * 1e-2
```

这与论文公式 `L_total = L_CD + 0.01 * L_transfer` 对应。注意日志里又乘了 `1e3` 或 `1e1`，那只是显示缩放，不是优化目标本身。

## 8. 解码器：`models/dec_net.py`

### 8.1 `MappingNet`

输入是点云 token 特征 `(B, 256, 192)`，通过多层全连接网络生成每个分支的隐式点特征。

### 8.2 `AXform`

它先预测 token 权重，再做加权汇聚：

```python
x_weights = softmax(...)
x = bmm(x_weights, x_base)
```

然后用 1D 卷积把聚合后的特征映射为 3D 坐标 `(B, 256, 3)`。

### 8.3 `Decoder_Network`

- `num_branch=8`：8 个独立的 MappingNet + AXform。
- 每个分支生成 256 个新点，共 2048 个新点。
- 再从输入 partial 中 FPS 取 1024 个点。
- 最终输出当前代码中的点数是 `8*256 + 1024 = 3072`。

论文把它描述为类似 XMFnet 的解码器；当前实现并不是“只生成缺失部分”，而是“生成新点后保留一部分观测点”。读代码时应特别注意这个点数与 `N_POINTS=2048` 不相等：`N_POINTS` 是输入采样数，不是 decoder 最终输出数。

## 9. 训练、验证、测试

### `run/train.py`

每个 batch 的真实顺序：

1. 取 `(view, gt_pc, part_pc)`。
2. CUDA 化。
3. 对 partial 和 GT 各做一次 FPS 到 2048 点。
4. `model(partial, png)` 得到 `recon, style_loss`。
5. 计算 `chamfer_sqrt(recon, gt)`。
6. 加入 `0.01 * style_loss`，反向传播并更新 Adam。
7. 每个 epoch 调用 `test_net()` 验证，并按 CD 保存 best checkpoint。

重点看这些变量：`partial`、`gt`、`png`、`recon`、`style_loss`、`cd`、`loss_total`。

### `run/test.py`

测试阶段：

- `model.eval()` + `torch.no_grad()`；
- 仍然对 partial 和 GT FPS 到 2048；
- 预测后调用 `calc_cd(..., calc_f1=True)`；
- 输出 `CDl1`、`CDl2`、`F1`。

指标对应关系：

- `CDl1`：对平方距离先开根号后平均，代码中的 `cd_p`；
- `CDl2`：平方距离直接平均，代码中的 `cd_t`；
- `F1`：阈值为 `0.001` 的近邻精度/召回调和。

## 10. CUDA 扩展与底层层：什么时候读

这些不是理解论文主线的第一优先级，但运行时报错时必须看：

| 文件 | 作用 |
|---|---|
| `models/pointnet2_batch/` | Ball Query、分组、FPS 等 PointNet++ CUDA 算子 |
| `models/layers/subsample.py` | FPS 和 gather 的 Python 包装 |
| `models/layers/group.py` | Ball Query/KNN 邻域分组 |
| `metrics/CD/chamfer3D/` | CUDA Chamfer Distance，导入时可能 JIT 编译 |
| `metrics/EMD/` | EMD 扩展；当前训练主路径使用 CD，不要先从 EMD 开始读 |
| `utils/furthestPointSampling/` | 训练/测试入口使用的 FPS 扩展 |

当前项目依赖 GPU；`loss_utils.py`、`subsample.py` 和 CUDA 扩展里有明确的 `torch.cuda` 路径，不能把“CPU 可运行”作为默认前提。

## 11. 最值得做的形状检查

建议在第一次理解代码时临时打印或用 hook 验证以下形状：

```text
view                 (B, 3, 224, 224)
gt / partial         (B, 2048, 3)
image tokens         (B, 256, 192)
point centers        (B, 256, 3)
point tokens         (B, 256, 192)
cross-attention out  (B, 256, 192)
each decoder branch  (B, 256, 3)
decoder output       (B, 3072, 3)
```

如果形状不符合，优先检查：`img_patch_size`、`pc_sample_rate`/`pc_sample_scale`、输入点数、`farthest_point_sample` 的 CUDA 扩展，以及 decoder 的 8 分支拼接。

## 12. 论文概念到代码的速查表

| 论文概念 | 当前代码位置 | 你要理解的事情 |
|---|---|---|
| View-guided completion | `ViPCDataLoader`, `EGIInet.forward` | partial + image -> complete point cloud |
| Image Tokenizer | `models/encoders.py:img_tokenizer` | 图像 patch -> 256 个 token |
| Point Cloud Tokenizer | `models/encoders.py:pc_tokenizer` | FPS + Ball Query -> 局部几何 token |
| Unified Encoder / SFE | `transfer_loss_shared_encoder.blocks` | 两种模态共用 ViT 结构 |
| Position embedding | `pc_pos_embed` | 把不规则点云中心坐标注入 token |
| SFTnet 的代码对应 | 最后 `fuse_layer_num` 个 block + FT-Loss | 当前实现没有独立 SFTnet 类，而是通过层间边界表达 |
| FT-Loss | `gram_matrix` 与 `style_transfer_loss` | 双向结构迁移 + 点云结构保持 |
| Feature Fusion | `cross_layer` | 点云 Query/Value、图像 Key；图像引导权重 |
| Completion Decoder | `models/dec_net.py` | 8 分支生成新点并拼接 partial FPS 点 |
| CD / F-score | `utils/loss_utils.py`, `run/test.py` | 训练目标与评估指标 |

## 13. 当前代码中需要保持警惕的实现事实

1. `run/train.py` 和 `run/test.py` 的 list 路径是硬编码的 Linux 路径；换机器先检查。
2. `config_vipc.py` 中的 `WEIGHTS` 当前非空；想从头训练时要确认不会误加载 checkpoint。
3. `view_align=False` 会随机选视角，所以同一个样本多次测试可能有轻微变化。
4. `Decoder_Network` 当前输出 3072 点，而输入/GT 在入口处采样为 2048 点。
5. `models/encoders.py` 的 `transfer_loss_shared_encoder` 名称容易让人误以为只有“共享编码”；实际上它同时包含 token 化、共享 Transformer、FT-Loss 相关中间特征和最终 cross-attention。
6. `loss_utils.py` 同时定义 CD、DCD、EMD 辅助函数，但训练主路径实际使用的是 `chamfer_sqrt`，不要把未调用函数当成当前训练目标。
7. `models/dec_net.py` 中的 `EdgePoolingLayer`、`PointNet` 等类目前不在 `EGIInet.forward()` 主路径上，第一次阅读可以跳过。

## 14. 建议的实际阅读/调试任务

按这个顺序做小实验，理解会比通读所有层文件更快：

1. 只实例化 `ViPCDataLoader`，打印一条样本的三种形状和视角编号。
2. 单独运行 `img_tokenizer`，验证 `224 -> 256 tokens`。
3. 单独运行 `pc_tokenizer`，打印每个 stage 的点数和特征通道。
4. 给 `transfer_loss_shared_encoder` 注册 hook，观察 `pc_f_list`、`im_f_list` 和 cross-attention 输出。
5. 单独运行 decoder，确认 8 个分支各生成 256 点、最终是 3072 点。
6. 用一个 batch 手算 `cd`、`style_loss`、`loss_total` 的数量级，确认 `0.01` 的权重确实生效。
7. 最后再运行 `python main.py --test` 或 1 epoch 训练。

如果你只想抓住论文创新点，优先读：

```text
ViPCdataloader.py
-> encoders.py: pc_tokenizer / transfer_loss_shared_encoder
-> loss_utils.py: calc_cd / chamfer_sqrt
-> dec_net.py: Decoder_Network
-> train.py: loss_total = cd + 0.01 * style_loss
```

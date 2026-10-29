"""根据查询坐标，计算残缺点云的 PCA 法向量或邻域点数密度。"""

import math
import numbers

import torch


class _SmallestEigenvector(torch.autograd.Function):
    """只对最小特征向量求一阶梯度，避免两个切向特征值相等时出现 0/0。

    平面邻域允许 l1=l2，只要求法向量对应的 l0 与其余特征值分离。
    不支持二阶梯度；特征值仅用于有效性判断，不参与求导。
    """

    @staticmethod
    def forward(ctx, covariance, eps):
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.eps = eps
        ctx.mark_non_differentiable(eigenvalues)
        return eigenvalues, eigenvectors[..., 0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_values, grad_normal):
        if grad_normal is None:
            return None, None
        eigenvalues, eigenvectors = ctx.saved_tensors
        normal = eigenvectors[..., 0]
        tangents = eigenvectors[..., 1:]
        gaps = eigenvalues[..., :1] - eigenvalues[..., 1:]
        tolerance = torch.maximum(
            eigenvalues[..., -1].abs() * (10 * torch.finfo(eigenvalues.dtype).eps),
            eigenvalues.new_tensor(ctx.eps),
        )
        separated = gaps.abs() > tolerance.unsqueeze(-1)
        safe_gaps = torch.where(separated, gaps, torch.ones_like(gaps))
        projected = (tangents.transpose(-1, -2) @ grad_normal.unsqueeze(-1)).squeeze(-1)
        coefficients = torch.where(separated, projected / safe_gaps, torch.zeros_like(projected))
        direction = (tangents @ coefficients.unsqueeze(-1)).squeeze(-1)
        # 对称协方差矩阵的梯度；仅出现 l0-l1 和 l0-l2，不除以 l1-l2。
        gradient = direction.unsqueeze(-1) * normal.unsqueeze(-2)
        return (gradient + gradient.transpose(-1, -2)) * .5, None


class PointCloudLocalGeometry:
    """提供法向量、密度计算，以及用于法向量估计的高密度中心筛选。

    Args:
        radius: 球邻域半径，使用半径内全部观测点；None 表示 KNN。
        k: KNN 模式的最近邻数量，仅在 radius=None 时生效。
        chunk_size: 每次处理的查询点数量，限制显存占用。
        eps: PCA 特征值的绝对退化阈值，单位为坐标单位的平方。

    两个方法都接收 points（观测点云）和 xyz（待查询坐标集合）。
    单云输入 points=[N,3]、xyz=[M,3]；批量输入为 [B,N,3]、[B,M,3]。
    xyz 省略时计算 points 中每个点的结果，输出与查询坐标顺序一致。
    中心点本身和重复采样点均计入邻域。

    返回 Tensor 的设备跟随 points；float64 保留双精度，其余转 float32。
    支持 CPU/CUDA，默认不保留梯度。compute_normals 可显式开启一阶求导。

    Example::

        geometry = PointCloudLocalGeometry(radius=0.1)
        points = torch.rand(2, 2048, 3, device='cuda')
        xyz = points[:, :128, :]
        normals = geometry.compute_normals(points, xyz)  # [2,128,3]
        density = geometry.compute_density(points, xyz)  # [2,128]
    """

    def __init__(self, radius=0.1, k=32, chunk_size=128, eps=1e-12):
        if radius is not None and (
            not isinstance(radius, numbers.Real)
            or not math.isfinite(radius) or radius <= 0
        ):
            raise ValueError("radius must be finite and positive, or None")
        for name, value in (("k", k), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.radius = radius
        self.k = k
        self.chunk_size = chunk_size
        self.eps = eps

    def compute_normals(self, points, xyz=None, differentiable=False):
        """PCA 法向量集合，返回 [M,3] 或 [B,M,3]，与 xyz 顺序一致。

        对每个查询点的邻居减去邻域质心，构造 3x3 协方差矩阵，取最小
        特征值对应的单位特征向量。为固定正负号，使绝对值最大的分量为正；
        此约定不保证法向量朝外。少于三个邻居、共线、重合或最小特征值
        无法区分的邻域返回零向量，并保留对应查询点的位置。

        differentiable=True 时保留相对于 points 的一阶梯度，用于生成点云
        的法向量 loss。半径筛选/KNN 索引视为固定邻域，不对邻居选择求导。
        在外部 torch.no_grad() 下仍不保留梯度。
        """
        with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
            return self._compute_normals(points, xyz)

    def _compute_normals(self, points, xyz):
        points, xyz, single = self._prepare_inputs(points, xyz)
        if xyz.shape[1] == 0:
            return xyz[0].clone() if single else xyz.clone()
        chunks = []
        for offsets, distances, selected in self._neighborhoods(points, xyz):
            count = selected.sum(dim=-1)
            denominator = count.clamp_min(1).to(points.dtype)
            offsets = torch.where(selected.unsqueeze(-1), offsets, torch.zeros_like(offsets))
            mean = offsets.sum(dim=2) / denominator.unsqueeze(-1)
            centered = offsets - mean.unsqueeze(2)
            centered = torch.where(selected.unsqueeze(-1), centered, torch.zeros_like(centered))
            covariance = centered.transpose(-1, -2).matmul(centered) / denominator[..., None, None]
            eigenvalues, normal = _SmallestEigenvector.apply(covariance, self.eps)
            l0, l1, l2 = eigenvalues.clamp_min(0).unbind(dim=-1)
            tolerance = torch.maximum(l2 * (10 * torch.finfo(points.dtype).eps), l2.new_tensor(self.eps))
            valid = (count >= 3) & (l1 > tolerance) & ((l1 - l0) > tolerance)
            orientation = normal.gather(-1, normal.abs().argmax(dim=-1, keepdim=True))
            normal = normal * torch.where(orientation < 0, -1., 1.)
            chunks.append(torch.where(valid.unsqueeze(-1), normal, torch.zeros_like(normal)))
        normals = torch.cat(chunks, dim=1)
        return normals[0] if single else normals

    @torch.no_grad()
    def compute_density(self, points, xyz=None):
        """邻域点数密度集合，返回 [M] 或 [B,M]，与 xyz 顺序一致。

        密度 = 邻居数 / (4*pi*r^3/3)。半径模式的 r 为指定半径，KNN
        模式的 r 为最远邻居距离。没有邻居或 r=0 时返回零。
        密度反映残缺点云的实际观测采样，不校正遮挡、缺失和边界。
        """
        points, xyz, single = self._prepare_inputs(points, xyz)
        if xyz.shape[1] == 0:
            density = xyz.new_empty(xyz.shape[:2])
            return density[0] if single else density
        chunks = []
        for offsets, distances, selected in self._neighborhoods(points, xyz):
            count = selected.sum(dim=-1).to(points.dtype)
            radius = (distances.max(dim=-1).values if self.radius is None
                      else torch.full_like(count, self.radius))
            positive = radius > 0
            safe_radius = torch.where(positive, radius, torch.ones_like(radius))
            density = count / ((4. / 3.) * math.pi * safe_radius ** 3)
            chunks.append(torch.where(positive, density, torch.zeros_like(density)))
        density = torch.cat(chunks, dim=1)
        return density[0] if single else density

    @torch.no_grad()
    def select_dense_centers(self, points, xyz, keep_ratio=0.25):
        """按邻域密度降序选取不重复的中心坐标。

        points 为原始观测点云，xyz 为候选中心，格式与 compute_normals 一致。
        目标数量 K=max(1,floor(M*keep_ratio))，M 为去重前的候选数量。
        同一云内坐标完全相同的中心仅保留一个，再从其余候选补足 K 个。
        密度在原始 points 中计算，使用与法向量计算相同的邻域参数。

        Returns:
            centers: [B,K,3] 或 [K,3]，按密度降序排列。
            density: [B,K] 或 [K]，与 centers 一一对应。
            valid: bool [B,K] 或 [K]，表示实际选中的中心。
                不同坐标不足 K 个时用零补齐，补齐位置为 False；
                不能通过坐标是否为零判断有效性，因为原点也可能是真实中心。
                此掩码只表示中心存在，PCA 是否有效还需检查法向量是否非零。
        """
        if (not isinstance(keep_ratio, numbers.Real) or not math.isfinite(keep_ratio)
                or not 0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be finite and in (0,1]")
        points, xyz, single = self._prepare_inputs(points, xyz)
        batch, num_centers, _ = xyz.shape
        keep = max(1, int(num_centers * keep_ratio)) if num_centers else 0
        centers = xyz.new_zeros((batch, keep, 3))
        density = xyz.new_zeros((batch, keep))
        valid = torch.zeros((batch, keep), dtype=torch.bool, device=xyz.device)
        if keep:
            candidate_density = self.compute_density(points, xyz)
            for b in range(batch):
                # 仅在同一个样本内去重，避免不同批次的坐标相互影响。
                unique_xyz, inverse = torch.unique(xyz[b], dim=0, return_inverse=True)
                # 相同坐标的邻域密度完全相同，映射到唯一坐标对应的分数。
                unique_density = candidate_density[b].new_zeros(unique_xyz.shape[0])
                unique_density.scatter_(0, inverse, candidate_density[b])
                count = min(keep, unique_xyz.shape[0])
                order = unique_density.argsort(descending=True, stable=True)[:count]
                centers[b, :count] = unique_xyz[order]
                density[b, :count] = unique_density[order]
                valid[b, :count] = True
        if single:
            return centers[0], density[0], valid[0]
        return centers, density, valid

    def _prepare_inputs(self, points, xyz):
        """统一输入的类型、设备和维度，供两个公开计算方法复用。

        points 是用于寻找邻居的观测点云，xyz 是需要计算结果的查询坐标。
        支持 Tensor、NumPy 数组或坐标列表；xyz=None 时查询全部观测点。

        Returns:
            points: 浮点 Tensor [B,N,3]，保留原 Tensor 的设备。
            xyz: 浮点 Tensor [B,M,3]，与 points 的设备和精度一致。
            single: 原始 points 是否为单云 [N,3]。调用方用它移除输出的
                临时批次维度，从而返回 [M,3] 法向量或 [M] 密度。

        此处不排序或采样，因此不会改变输入坐标的顺序。
        """
        # 将列表/数组转为 Tensor；已有 Tensor 尽可能复用原数据和设备。
        points = torch.as_tensor(points)
        if points.is_complex():
            raise ValueError("points must contain real coordinates")
        # 保留双精度，其余类型统一为 float32，供距离与特征值计算使用。
        dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
        points = points.to(dtype=dtype)
        # 单云 [N,3] 临时变为 [1,N,3]，让后续算法统一按批量形式处理。
        single = points.ndim == 2
        if single:
            points = points.unsqueeze(0)
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("points must have shape [N,3] or [B,N,3], with B,N > 0")
        if xyz is None:
            # 未指定查询坐标时，对观测点云中的每个点计算结果。
            xyz = points
        else:
            # 查询点跟随观测点云的设备；points 在 CUDA 上时就在 GPU 计算。
            xyz = torch.as_tensor(xyz, device=points.device)
            if xyz.is_complex():
                raise ValueError("xyz must contain real coordinates")
            xyz = xyz.to(dtype=dtype)
            if single:
                # 单个查询 [3] -> [1,3] -> [1,1,3]；
                # 多个查询 [M,3] -> [1,M,3]。
                if xyz.ndim == 1:
                    xyz = xyz.unsqueeze(0)
                xyz = xyz.unsqueeze(0)
        # 每个批次的查询点只能在对应批次的观测点云中寻找邻居。
        if xyz.ndim != 3 or xyz.shape[0] != points.shape[0] or xyz.shape[-1] != 3:
            raise ValueError("xyz must be [3]/[M,3] for one cloud, or [B,M,3]")
        # 拒绝 NaN 和正负无穷，避免无效距离或协方差影响计算。
        if not torch.isfinite(points).all() or not torch.isfinite(xyz).all():
            raise ValueError("points and xyz must contain finite coordinates")
        return points, xyz, single

    def _neighborhoods(self, points, xyz):
        """按查询点顺序分块生成邻域，供法向量和密度计算共用。

        输入已由 _prepare_inputs 统一为 points=[B,N,3]、xyz=[B,M,3]。
        每块处理 C 个查询点，C <= chunk_size，避免一次生成全部查询点
        对应的 [B,M,N,3] 相对坐标张量。所有张量运算跟随输入设备。

        Yields:
            offsets: [B,C,L,3]，候选观测点减去查询点的相对坐标。
            distances: [B,C,L]，上述相对坐标的欧氏距离。
            selected: bool [B,C,L]，标记哪些候选点属于邻域。

        半径模式 L=N，保留全部候选点，通过 selected 标记球内点；
        球外点仍在 offsets/distances 中，调用方必须使用掩码筛选。
        KNN 模式 L=min(k,N)，只保留最近邻，因此 selected 全为 True。
        """
        # 连续切分 xyz，不改变查询顺序；调用方沿查询维拼接即可恢复顺序。
        for start in range(0, xyz.shape[1], self.chunk_size):
            query = xyz[:, start:start + self.chunk_size]
            # 以查询坐标为原点，减少较大平移造成的协方差消减误差。
            # 广播：[B,1,N,3] - [B,C,1,3] -> [B,C,N,3]。
            offsets = points[:, None, :, :] - query[:, :, None, :]
            # 对最后的 x/y/z 三个分量求二范数，得到 [B,C,N] 距离。
            distances = torch.linalg.vector_norm(offsets, dim=-1)
            if self.radius is None:
                # KNN：沿观测点维度选择最近的 k 个点，点数不足时取全部。
                distances, indices = distances.topk(min(self.k, points.shape[1]), largest=False)
                # 用相同索引提取相对坐标，保证它与筛选后的距离一一对应。
                offsets = offsets.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 3))
                selected = torch.ones_like(distances, dtype=torch.bool)
            else:
                # 球邻域：包含距离恰好等于半径的点，不受 k 限制。
                selected = distances <= self.radius
            # yield 每次交出一块邻域；下次迭代再继续处理下一块查询点。
            yield offsets, distances, selected

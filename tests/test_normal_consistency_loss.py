"""验证真实法向量损失与梯度，隔离 loss_utils 导入时无关的 CUDA 扩展编译。"""

import ast
import math
from pathlib import Path
import unittest

import torch

from utils.point_cloud_geometry import PointCloudLocalGeometry


# 直接加载实际函数定义；Chamfer/EMD 不参与这些测试，也不替代几何计算。
source_path = Path(__file__).resolve().parents[1] / "utils" / "loss_utils.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "normal_consistency_loss")
namespace = {"torch": torch, "PointCloudLocalGeometry": PointCloudLocalGeometry}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
normal_consistency_loss = namespace["normal_consistency_loss"]


class NormalConsistencyLossTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PointCloudLocalGeometry(radius=4)
        # 各向同性平面采样：两个切向特征值相等，但法向量仍然唯一。
        self.plane = torch.tensor([
            [-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.], [1., 1., 0.],
        ], dtype=torch.float64)
        self.centers = [torch.zeros(1, 3, dtype=torch.float64)]
        self.targets = [torch.tensor([[0., 0., 1.]], dtype=torch.float64)]

    def tilted(self, angle=.4):
        rotation = torch.tensor([
            [math.cos(angle), 0., -math.sin(angle)],
            [0., 1., 0.], [math.sin(angle), 0., math.cos(angle)],
        ], dtype=torch.float64)
        return (self.plane @ rotation.T).unsqueeze(0)

    def test_matching_and_opposite_normals(self):
        recon = self.plane.unsqueeze(0).requires_grad_(True)
        for targets in (self.targets, [-self.targets[0]]):
            loss = normal_consistency_loss(recon, self.centers, targets, self.geometry)
            self.assertAlmostEqual(loss.item(), 0.)
            gradient = torch.autograd.grad(loss, recon)[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_expected_angle_and_gradient(self):
        recon = self.tilted().requires_grad_(True)
        centers = [self.centers[0].clone().requires_grad_(True)]
        targets = [self.targets[0].clone().requires_grad_(True)]
        loss = normal_consistency_loss(recon, centers, targets, self.geometry)
        self.assertAlmostEqual(loss.item(), 1 - math.cos(.4))
        loss.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())
        self.assertGreater(recon.grad.abs().sum().item(), 0)
        self.assertIsNone(centers[0].grad)
        self.assertIsNone(targets[0].grad)
        improved = (recon.detach() - .1 * recon.grad).requires_grad_(True)
        updated = normal_consistency_loss(improved, centers, targets, self.geometry)
        self.assertLess(updated.item(), loss.item())

    def test_gradient_matches_finite_differences(self):
        recon = self.tilted().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda points: normal_consistency_loss(points, self.centers, self.targets, self.geometry),
            (recon,), eps=1e-6, atol=1e-5, rtol=1e-3,
        ))

    def test_all_invalid_and_empty_targets(self):
        for points in (torch.zeros(1, 4, 3, dtype=torch.float64),
                       torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]], dtype=torch.float64)):
            for centers, targets in ((self.centers, self.targets),
                                     ([torch.empty(0, 3)], [torch.empty(0, 3)])):
                recon = points.clone().requires_grad_(True)
                loss = normal_consistency_loss(recon, centers, targets, self.geometry)
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(recon.grad).all())
                self.assertEqual(recon.grad.abs().sum().item(), 0)

    def test_invalid_target_and_missing_neighborhood(self):
        recon = self.tilted().requires_grad_(True)
        centers = [torch.tensor([[0., 0., 0.], [0., 0., 0.], [100., 100., 100.]], dtype=torch.float64)]
        targets = [torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 1.]], dtype=torch.float64)]
        loss = normal_consistency_loss(recon, centers, targets, self.geometry)
        self.assertAlmostEqual(loss.item(), 1 - math.cos(.4))
        loss.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())

    def test_batch_reduction_and_variable_lengths(self):
        recon = torch.cat([self.tilted(), self.tilted(.2), self.tilted(.7)])
        centers = [self.centers[0].repeat(2, 1), self.centers[0], torch.empty(0, 3)]
        targets = [self.targets[0].repeat(2, 1), self.targets[0], torch.empty(0, 3)]
        loss = normal_consistency_loss(recon, centers, targets, self.geometry)
        self.assertAlmostEqual(loss.item(), (2 * (1 - math.cos(.4)) + 1 - math.cos(.2)) / 3)

    def test_gradient_is_opt_in(self):
        recon = self.tilted()[0].requires_grad_(True)
        self.assertFalse(self.geometry.compute_normals(recon).requires_grad)
        self.assertTrue(self.geometry.compute_normals(recon, differentiable=True).requires_grad)
        with torch.no_grad():
            self.assertFalse(self.geometry.compute_normals(recon, differentiable=True).requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_loss_and_gradient_match_cpu(self):
        for dtype in (torch.float32, torch.float64):
            values, gradients = [], []
            for device in ("cpu", "cuda"):
                recon = self.tilted().to(device=device, dtype=dtype).requires_grad_(True)
                loss = normal_consistency_loss(recon, self.centers, self.targets, self.geometry)
                loss.backward()
                self.assertTrue(torch.isfinite(recon.grad).all())
                self.assertGreater(recon.grad.abs().sum().item(), 0)
                values.append(loss.detach().cpu())
                gradients.append(recon.grad.cpu())
            torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-4)
            torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

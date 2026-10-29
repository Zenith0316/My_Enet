"""验证解码器保留有效中心并补足观测点；用参考 FPS 隔离 CUDA 扩展依赖。"""

import ast
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F


def reference_fps(points, count):
    """用于测试的确定性 FPS，与生产调用的 CUDA FPS 具有相同接口。"""
    batch, size, _ = points.shape
    indices = torch.empty(batch, count, dtype=torch.long, device=points.device)
    distances = torch.full((batch, size), float("inf"), device=points.device, dtype=points.dtype)
    farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for i in range(count):
        indices[:, i] = farthest
        center = points[rows, farthest].unsqueeze(1)
        distances = torch.minimum(distances, (points - center).square().sum(dim=-1))
        farthest = distances.argmax(dim=-1)
    return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))


def load_decoder(fps):
    # 执行实际类定义；只替代原有 FPS 扩展，避免导入时编译无关的 Chamfer/EMD。
    path = Path(__file__).resolve().parents[1] / "models" / "dec_net.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("MappingNet", "AXform", "Decoder_Network")]
    namespace = {"torch": torch, "nn": nn, "F": F, "farthest_point_sample": fps}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Decoder_Network"]


class DecoderPartialSamplingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fps(points, count):
            self.calls.append(count)
            return reference_fps(points, count)

        self.decoder = load_decoder(fps)(K1=8, K2=8, N=8)
        self.points = torch.arange(36, dtype=torch.float32).reshape(1, 12, 3)

    def test_variable_valid_counts_and_excluded_centers(self):
        points = self.points.expand(2, -1, -1).clone()
        centers = points[:, [8, 2, 6]].clone()
        mask = torch.tensor([[True, False, False], [True, True, False]])
        result = self.decoder._sample_partial(points, centers, mask, num_points=6)
        self.assertEqual(result.shape, (2, 6, 3))
        self.assertEqual(self.calls, [5, 4])
        for b, count in ((0, 1), (1, 2)):
            torch.testing.assert_close(result[b, :count], centers[b, mask[b]])
            self.assertEqual(torch.unique(result[b], dim=0).shape[0], 6)
            self.assertTrue((result[b, :, None] == points[b, None]).all(dim=-1).any(dim=-1).all())

    def test_duplicate_centers_and_candidates(self):
        points = torch.cat([self.points, self.points[:, :5]], dim=1)
        centers = points[:, [3, 3, 0]]
        result = self.decoder._sample_partial(points, centers, torch.ones(1, 3, dtype=torch.bool), 6)
        torch.testing.assert_close(result[0, :2], points[0, [3, 0]])
        self.assertEqual(self.calls, [4])
        self.assertEqual(torch.unique(result[0], dim=0).shape[0], 6)

    def test_no_valid_centers_and_legacy_call(self):
        centers = torch.full((1, 2, 3), float("nan"))
        result = self.decoder._sample_partial(self.points, centers, torch.zeros(1, 2, dtype=torch.bool), 6)
        self.assertEqual(torch.unique(result[0], dim=0).shape[0], 6)
        legacy = self.decoder._sample_partial(self.points, num_points=6)
        torch.testing.assert_close(legacy, reference_fps(self.points, 6))

    def test_insufficient_unique_points_repeat_to_budget(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]]])
        centers = torch.tensor([[[0., 0., 0.]]])
        result = self.decoder._sample_partial(points, centers, torch.ones(1, 1, dtype=torch.bool), 6)
        self.assertEqual(result.shape, (1, 6, 3))
        torch.testing.assert_close(result[0, :2], points[0, :2])
        self.assertEqual(len(self.calls), 0)

    def test_full_center_budget_and_overflow(self):
        centers = self.points[:, :6]
        mask = torch.ones(1, 6, dtype=torch.bool)
        torch.testing.assert_close(self.decoder._sample_partial(self.points, centers, mask, 6), centers)
        with self.assertRaises(ValueError):
            self.decoder._sample_partial(self.points, centers, mask, 5)

    def test_real_decoder_output_and_generated_point_gradients(self):
        # 使用真实分支网络；不足 1024 个不同观测点时补齐，最终仍输出 2048 点。
        self.decoder.eval()
        features = torch.randn(1, 16, 8, requires_grad=True)
        centers = self.points[:, [3, 7]]
        mask = torch.ones(1, 2, dtype=torch.bool)
        output = self.decoder(features, self.points, centers, mask)
        self.assertEqual(output.shape, (1, 2048, 3))
        torch.testing.assert_close(output[:, 1024:1026], centers)
        output[:, :1024].square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_sampling_matches_cpu(self):
        centers = self.points[:, [2, 7]]
        mask = torch.ones(1, 2, dtype=torch.bool)
        cpu = self.decoder._sample_partial(self.points, centers, mask, 6)
        gpu = self.decoder._sample_partial(self.points.cuda(), centers.cuda(), mask.cuda(), 6)
        self.assertEqual(gpu.device.type, "cuda")
        torch.testing.assert_close(cpu, gpu.cpu())


if __name__ == "__main__":
    unittest.main()

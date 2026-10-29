import math
import unittest

import torch

from utils.point_cloud_geometry import PointCloudLocalGeometry


class PointCloudLocalGeometryTests(unittest.TestCase):
    def setUp(self):
        self.plane = torch.tensor([
            [-1., -1., 0.], [-1., 1., 0.],
            [1., -1., 0.], [1., 1., 0.], [0., 0., 0.],
        ], dtype=torch.float64)

    def test_order_and_repeated_queries(self):
        vertical = torch.tensor([
            [10., -1., -1.], [10., -1., 1.], [10., 1., -1.], [10., 1., 1.],
        ], dtype=torch.float64)
        points = torch.cat([self.plane, vertical])
        xyz = [[10, 0, 0], [0, 0, 0], [100, 100, 100], [10, 0, 0]]
        util = PointCloudLocalGeometry(radius=2, chunk_size=1)
        normals = util.compute_normals(points, xyz)
        expected = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(normals, expected)
        expected_density = torch.tensor([4., 5., 0., 4.], dtype=torch.float64) / (4 / 3 * math.pi * 8)
        torch.testing.assert_close(util.compute_density(points, xyz), expected_density)

    def test_default_queries_and_batch(self):
        util = PointCloudLocalGeometry(radius=4)
        points = torch.stack([self.plane, self.plane + 10])
        normals = util.compute_normals(points)
        density = util.compute_density(points)
        self.assertEqual(normals.shape, (2, 5, 3))
        self.assertEqual(density.shape, (2, 5))
        expected = torch.zeros_like(points)
        expected[..., 2] = 1
        torch.testing.assert_close(normals, expected)
        torch.testing.assert_close(density, torch.full((2, 5), 5 / (4 / 3 * math.pi * 64), dtype=torch.float64))
        torch.testing.assert_close(util.compute_normals(self.plane), normals[0])
        torch.testing.assert_close(util.compute_density(self.plane), density[0])

    def test_radius_boundary(self):
        points = [[0., 0., 0.], [1., 0., 0.], [1.01, 0., 0.]]
        util = PointCloudLocalGeometry(radius=1, k=1)
        self.assertAlmostEqual(util.compute_density(points, [0, 0, 0]).item(), 2 / (4 / 3 * math.pi), places=6)

    def test_knn_and_degenerate_points(self):
        util = PointCloudLocalGeometry(radius=None, k=20)
        torch.testing.assert_close(util.compute_normals(self.plane, [0, 0, 0]),
                                   torch.tensor([[0., 0., 1.]], dtype=torch.float64))
        self.assertAlmostEqual(util.compute_density(self.plane, [0, 0, 0]).item(), 5 / (4 / 3 * math.pi * math.sqrt(2) ** 3))
        for points in (torch.zeros(4, 3), torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                       torch.eye(3)[:2], torch.cat([torch.eye(3), -torch.eye(3)])):
            normal = util.compute_normals(points, [0, 0, 0])
            self.assertEqual(normal.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(util.compute_density(points, [0, 0, 0])).all())
        self.assertEqual(util.compute_density(torch.zeros(4, 3), [0, 0, 0]).item(), 0)
        self.assertEqual(PointCloudLocalGeometry(radius=None, k=2).compute_normals(self.plane, [0, 0, 0]).abs().sum().item(), 0)

    def test_sloped_plane_and_translation(self):
        points = self.plane.clone()
        points[:, 2] = points[:, 0] + 2 * points[:, 1]
        shift = torch.tensor([10000., -3000., 2000.], dtype=torch.float64)
        util = PointCloudLocalGeometry(radius=10)
        # 最大绝对值分量为正，方向取 [1,2,-1]。
        expected = torch.tensor([[1., 2., -1.]], dtype=torch.float64) / math.sqrt(6)
        torch.testing.assert_close(util.compute_normals(points + shift, shift), expected)

    def test_chunking_and_no_grad(self):
        points = self.plane.clone().requires_grad_(True)
        xyz = torch.tensor([[0., 0., 0.], [.3, .3, 0.], [2., 0., 0.]])
        for radius in (None, 1.5):
            small = PointCloudLocalGeometry(radius=radius, chunk_size=1)
            large = PointCloudLocalGeometry(radius=radius, chunk_size=10)
            for name in ("compute_normals", "compute_density"):
                result = getattr(small, name)(points, xyz)
                torch.testing.assert_close(result, getattr(large, name)(points, xyz))
                self.assertFalse(result.requires_grad)

    def test_empty_queries(self):
        util = PointCloudLocalGeometry()
        self.assertEqual(util.compute_normals(self.plane, torch.empty(0, 3)).shape, (0, 3))
        self.assertEqual(util.compute_density(self.plane, torch.empty(0, 3)).shape, (0,))

    def test_dense_selection_deduplicates_and_refills(self):
        # 四个相隔很远的邻域，分别有 5、4、3、1 个不同的观测点。
        points = torch.cat([
            self.plane * .05,
            self.plane[:4] * .05 + torch.tensor([10., 0., 0.]),
            self.plane[:3] * .05 + torch.tensor([20., 0., 0.]),
            torch.tensor([[30., 0., 0.]], dtype=torch.float64),
        ])
        # 最高密度候选有多个重复；去重后应继续选第二、第三个邻域。
        xyz = torch.tensor([[30., 0., 0.]] * 2 + [[0., 0., 0.]] * 5
                           + [[20., 0., 0.]] * 2 + [[10., 0., 0.]] * 3, dtype=torch.float64)
        util = PointCloudLocalGeometry(radius=.2)
        centers, density, valid = util.select_dense_centers(points, xyz)
        expected = torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(centers, expected)
        torch.testing.assert_close(density, torch.tensor([5., 4., 3.], dtype=torch.float64) / (4 / 3 * math.pi * .2 ** 3))
        self.assertTrue(valid.all())
        normals = util.compute_normals(points, centers)
        torch.testing.assert_close(normals, torch.tensor([[0., 0., 1.]] * 3, dtype=torch.float64))

    def test_dense_selection_batch_padding_and_empty(self):
        util = PointCloudLocalGeometry(radius=2)
        points = torch.stack([self.plane, self.plane + 10])
        # 原点是真实中心，不能把零坐标本身当成补齐标志。
        xyz = torch.zeros(2, 12, 3, dtype=torch.float64)
        xyz[1, :6] = 10
        xyz[1, 6:] = 11
        centers, density, valid = util.select_dense_centers(points, xyz)
        self.assertEqual(centers.shape, (2, 3, 3))
        self.assertEqual(valid.sum(dim=1).tolist(), [1, 2])
        self.assertEqual(density[~valid].abs().sum().item(), 0)
        self.assertTrue((centers[0, 0] == 0).all())
        for size in (0, 1, 3):
            selected, scores, mask = util.select_dense_centers(self.plane, self.plane[:size])
            self.assertEqual(selected.shape, (min(1, size), 3))
            self.assertEqual(scores.shape, mask.shape)
        for ratio in (0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                util.select_dense_centers(self.plane, self.plane, keep_ratio=ratio)

    def test_validation(self):
        for kwargs in ({"radius": 0}, {"radius": float("nan")}, {"k": 0},
                       {"k": 1.5}, {"chunk_size": 0}, {"eps": 0}):
            with self.assertRaises(ValueError):
                PointCloudLocalGeometry(**kwargs)
        util = PointCloudLocalGeometry()
        for method in (util.compute_normals, util.compute_density):
            for points, xyz in ((torch.empty(0, 3), [0, 0, 0]),
                                (self.plane, [float("nan"), 0, 0]),
                                (torch.full((3, 3), float("nan")), [0, 0, 0]),
                                (self.plane, torch.zeros(2, 3, 3))):
                with self.assertRaises(ValueError):
                    method(points, xyz)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        xyz = torch.tensor([[.1, .2, 0.], [10., 10., 10.]], dtype=torch.float64)
        for radius in (None, 2.):
            for dtype in (torch.float32, torch.float64):
                util = PointCloudLocalGeometry(radius=radius)
                points = self.plane.to(dtype)
                for method in (util.compute_normals, util.compute_density):
                    gpu = method(points.cuda(), xyz.cuda())
                    self.assertEqual(gpu.device.type, "cuda")
                    self.assertEqual(gpu.dtype, dtype)
                    torch.testing.assert_close(method(points, xyz), gpu.cpu())
                cpu_selected = util.select_dense_centers(points, points, keep_ratio=.5)
                gpu_selected = util.select_dense_centers(points.cuda(), points.cuda(), keep_ratio=.5)
                for cpu_value, gpu_value in zip(cpu_selected, gpu_selected):
                    self.assertEqual(gpu_value.device.type, "cuda")
                    torch.testing.assert_close(cpu_value, gpu_value.cpu())


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from types import SimpleNamespace

import runtime_paths


def test_relative_override_uses_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VIPC_DATA', 'custom-data')
    assert runtime_paths.env_path('VIPC_DATA', 'unused') == str(
        (runtime_paths.PROJECT_ROOT / 'custom-data').resolve())


def test_archive_defaults_on_linux(monkeypatch):
    monkeypatch.setattr(runtime_paths, 'os', SimpleNamespace(name='posix'))
    assert runtime_paths.storage_default('runs', 'runs') == Path('/home4/zt26/runs')
    assert runtime_paths.storage_default('data/raw/shapenet_vipc/checkpoint', 'checkpoints') == Path(
        '/home4/zt26/data/raw/shapenet_vipc/checkpoint')


def test_dataset_link_and_explicit_override_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_paths, 'PROJECT_ROOT', tmp_path)
    linked = tmp_path / 'data/shapenetvipc'
    linked.mkdir(parents=True)
    assert runtime_paths.vipc_default() == linked
    override = tmp_path / 'another-dataset'
    monkeypatch.setenv('VIPC_DATA', str(override))
    assert runtime_paths.env_path('VIPC_DATA', runtime_paths.vipc_default()) == str(override.resolve())


def test_dataset_archive_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_paths, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(runtime_paths, 'os', SimpleNamespace(name='posix'))
    assert runtime_paths.vipc_default() == Path('/home4/zt26/data/raw/shapenet_vipc/ShapeNetViPC-Dataset')

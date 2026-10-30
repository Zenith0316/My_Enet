"""Portable paths; relative overrides are resolved against the repository root."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_HOME = Path('/home4/zt26')


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def env_path(name, default):
    return project_path(os.environ.get(name) or default)


def storage_default(server_relative, local_relative):
    """Linux 使用服务器归档目录；Windows 保留项目内的本地目录。"""
    if os.name == 'nt':
        return PROJECT_ROOT / local_relative
    return SERVER_HOME / server_relative


def vipc_default():
    """优先使用归档约定的项目软链接，不扫描数据目录。"""
    linked_data = PROJECT_ROOT / 'data' / 'shapenetvipc'
    if linked_data.is_dir():
        return linked_data
    return storage_default('data/raw/shapenet_vipc/ShapeNetViPC-Dataset',
                           'datasets/ShapeNetViPC-Dataset')

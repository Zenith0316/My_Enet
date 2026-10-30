"""Portable paths; relative overrides are resolved against the repository root."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def env_path(name, default):
    return project_path(os.environ.get(name) or default)

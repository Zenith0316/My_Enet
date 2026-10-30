#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# For Slurm, prefix this script with srun and your cluster resource options.
"${PYTHON:-python}" -m pip install .

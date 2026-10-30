#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Only remove this extension's local build products.
rm -rf -- __pycache__ dist build emd.egg-info

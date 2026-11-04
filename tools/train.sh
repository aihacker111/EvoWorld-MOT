#!/usr/bin/env bash
# Single-process training with compact console output; remaining args go to evoworld train.
set -euo pipefail
evoworld_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evoworld_python=${EVOWORLD_PYTHON:-python}
exec "$evoworld_python" -u "$evoworld_repo/tools/run_training.py" "$@"

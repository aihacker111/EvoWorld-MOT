#!/usr/bin/env bash
# One CUDA GPU, one process; defaults: batch 2 * 8 accumulation = 16 clips per update,
# the same effective batch as tools/train_2gpu.sh, so learning rates keep their meaning.
# Any flag after the defaults overrides them (e.g. --config, --batch-size, --out).
#   EVOWORLD_GPU=1 bash tools/train_1gpu.sh ...   # pick another visible GPU (default 0)
set -euo pipefail

evoworld_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evoworld_python=${EVOWORLD_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${EVOWORLD_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
cd "$evoworld_repo"

"$evoworld_python" - <<'PY'
import os
import torch
if not torch.cuda.is_available():
    raise SystemExit('This launcher needs one visible CUDA GPU. Enable a GPU before launching.')
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f'Using CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}: {name} ({memory:.1f} GiB); '
      'effective batch is reported below after CLI overrides.', flush=True)
PY

exec "$evoworld_python" -u "$evoworld_repo/tools/run_training.py" \
  --config configs/full.yaml --batch-size 2 --grad-accumulation 8 --workers 8 "$@"

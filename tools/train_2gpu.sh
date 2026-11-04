#!/usr/bin/env bash
# One process per GPU, effective batch 1 * 2 accumulation * 2 GPUs = 4 clips.
set -euo pipefail

evoworld_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evoworld_python=${EVOWORLD_PYTHON:-python}
evoworld_port=${EVOWORLD_MASTER_PORT:-29517}
cd "$evoworld_repo"

"$evoworld_python" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit('This launcher needs two visible CUDA GPUs. Enable both GPUs before launching.')
print('Using CUDA GPUs 0 and 1; effective batch=4 clips by default.', flush=True)
PY

exec "$evoworld_python" -u "$evoworld_repo/tools/run_training.py" \
  --distributed --master-port "$evoworld_port" \
  --config configs/full.yaml --batch-size 8 --grad-accumulation 2 --workers 1 "$@"

# DDP and AMP cache fix validation — 2026-11-04

- Default full suite: **41 passed, 1 skipped in 10.05s**. The skip is the opt-in localhost distributed integration test.
- Explicit `RUN_DISTRIBUTED_TESTS=1 pytest tests/test_distributed.py -q`: **3 passed in 52.07s**.
- Ruff check/format and `bash -n tools/train_2gpu.sh`: passed.

The real two-process test uses torchrun with CPU/Gloo and a synthetic tiny-perception model. It checks parameters and global AL counters are identical on both ranks. It also checks uninterrupted training against stop/resume, including rank-local RNG and DataLoader spawn: every saved model tensor matches exactly. Seven clips produce six usable clips and one reported tail drop, avoiding duplicate padding. Conditional accumulation is checked analytically: a parameter used only on rank0's earlier no_sync microbatch receives its expected averaged gradient (.25) on both ranks.

Unit tests cover disjoint sampler partitions/global resume offsets and synchronized AMP skip/backoff when a remote rank overflows but local gradients are finite. Actual CUDA GradScaler/NCCL convergence has not been tested on the development Mac.

The reported CUDA error came from masked assignment of Half visibility into a Float affinity buffer. Forecast summary and affinity calculations now explicitly use FP32, including half-cache reductions and gate casts. FP16 and BF16 cache tests verify finite FP32 outputs and nonzero finite gradients back to a low-precision gate. Perception/dynamics can continue using autocast; AMP is not globally disabled.

The checkpoint remains format2 with optional distributed metadata, world size and per-rank RNG. Inference still reads earlier checkpoints. Exact resume requires the same topology, source/config provenance and unchanged data. Transitioning the user's earlier single-GPU run to this updated two-GPU source uses explicit weights-only `--init-checkpoint` into a new run; optimizer/sampler history restarts.

The launcher defaults to batch1/GPU × accumulation2 × 2 processes = effective batch4, preserving the user's previous effective batch. No automatic LR scaling is applied. Only rank0 writes/validates; loss/counters are globally reduced, replay probe density is preserved per global microbatch, and extra replay ownership rotates across ranks.

This validation establishes CPU multi-process correctness and the precision-conversion regression. It does not establish actual 2-GPU speedup, CUDA bitwise reproducibility, peak VRAM, or final tracking quality. Verify startup `world_size=2`, devices cuda:0/cuda:1 and effective_batch=4 on Kaggle before a long run.

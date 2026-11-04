Latest DDP/AMP checks: [DISTRIBUTED_VALIDATION.md](DISTRIBUTED_VALIDATION.md).

# AL v0.2 validation — 2026-11-04

Final automated suite: **29 passed in 13.34 seconds**, development Mac CPU. Ruff check and formatting passed. These are mechanism/integration checks, not evidence of trained real-world tracking accuracy or confirmed novelty.

## Coverage

- Real Falcon strict loading of 493 canonical tensors, 10 classes; finite full-objective backward reaches YOLO's first convolution and all trainable module groups, including AL.
- Per-track tentative confirmation, active/dormant transitions, uncertainty growth during occlusion, reduced uncertainty on observation, immutable-ID recovery and no memory writes on missing observations.
- One-to-one global arbitration, agent-specific null choices, ghost tentative expiry, learned birth rejection, learned termination and protection of current trusted measurements against a terminal-policy error.
- Global budget/agent-credit enforcement, actual ROI acquisition accounting, zero-budget operation, real-image pooled features and shared physical forecasts across candidate memory actions.
- Repair gate receives the correct training-gradient direction; a controlled policy executes repair of a deliberately corrupted working memory using a real measurement and protected anchor.
- Sequence-aware alive labels bridge internal missing GT and censor insufficiently observed sequence tails. Corruption/dropout preserve GT and anchors.
- Paired intervention replay is isolated; correct/swap/write/freeze/wait examples train effects. Actor-only backward leaves critic parameters without gradients.
- v2 CPU interrupted/resumed optimization reproduces model tensors exactly; v1 inference dispatches to legacy tracking and explicit warm-start transfers common weights with fresh AL parameters.
- Simulated AMP overflow tests skip counters for unoptimized examples. This tests control flow; CUDA kernels/FP16 convergence were not tested on this Mac.
- Existing camera geometry, preprocessing, data validation, IDF1/class/ignore handling and codec fallback regressions remain covered.

Some mechanism tests deliberately set policy-head biases to trigger particular choices. They verify that learned outputs control the loop, not that a model trained on real data already makes accurate choices.

## Integration smoke

Supplied detector + small latent configuration trained for two complete optimizer updates on deterministic 96×64 synthetic images. All losses/gradients finite. Checkpoint format2 reloads strictly with 48 optimized policy examples and 18 critic examples. See `al_smoke_metrics.jsonl`.

Streaming inference wrote MOT/class files, an AVI preview decoded as 18 frames, and 18 `life.jsonl` frame records. All recorded extra compute stayed within budget. TrackEval1.3.0 evaluation completed, including the empty-prediction case. The aerial detector did not detect the colored synthetic rectangles; HOTA/IDF1 zero here is a pipeline edge-case check, not quality evidence.

Full configuration parameters:

| Component | Parameters |
|---|---:|
| Perception + ReID | 7,589,870 |
| Object world model | 448,464 |
| Camera | 40,716 |
| Association/gate/legacy critic | 53,765 |
| Per-track AL controller | 76,939 |
| **Total** | **8,209,754** |

The existing system environment emits a torchvision compatibility warning and its OpenCV build lacks MP4 encoding; isolated temporary Ultralytics/TrackEval installs and AVI fallback were used without changing the base environment. Install the clean compatible environment in README for actual training. No real-data convergence, full-resolution VRAM/GPU benchmark, CUDA bitwise reproducibility or official VisDrone result is claimed. Historical v1 validation remains in `VALIDATION_V1.md`.

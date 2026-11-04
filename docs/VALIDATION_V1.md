# Historical v1 validation

# Validation performed on 2026-11-04

These checks validate implementation behavior. They do not establish tracking quality, convergence, paper novelty, or GPU throughput.

- Final test suite: **15 passed in 20.22 seconds** on the development Mac CPU.
- `ruff check src tests tools`: passed; code formatted with Ruff.
- Supplied Falcon detector: strict load of 493 canonical tensors, 10 classes, epoch 120; original checkpoint SHA256 verified.
- Real detector joint forward/backward: finite, nonzero gradient reaches first YOLO convolution; full objective also reaches world, camera, association and memory-gate modules.
- Curriculum path: actual YOLO proposals decoded from the existing pyramid without modifying raw loss inputs; missed-proposal frames backpropagate correctly.
- Deterministic CPU resume: uninterrupted versus interrupted/resumed training produce identical model tensors at three optimizer updates with accumulation=2. An already-reached total-update limit performs no extra update.
- Tracker checks: births, empty detections, missing frames, immutable IDs within each hypothesis, class gating, one-to-one assignment, reset, persistent distinct descendants and bounded population.
- Counterfactual replay: paired inputs remain unmodified; replay results are reproducible; forced identity swaps produce identity errors; gate endpoints preserve or replace memory as specified.
- Geometry: camera affine transformation and multi-mode box envelopes checked numerically.
- Data/metrics: deterministic ordered clips and augmentation, duplicate identity rejection, RGB normalization, class-aware IDF1/ID-switch calculations and explicit ignore-region handling.
- Real-detector synthetic smoke: two complete optimizer updates, finite losses and gradients, effect examples increased to 24. This uses 96×64 synthetic images and reduced latent dimensions, not the full training configuration.
- Streaming/export: 18 synthetic frames processed; MOT/class outputs written; AVI preview opened and decoded with 18 frames at 96×64. This OpenCV build lacks an MP4 encoder; fallback to MJPEG/AVI was exercised and regression-tested.
- TrackEval 1.3.0: perfect-identity HOTA/AssA/DetA all equal 1.0 in a controlled metric check. End-to-end empty-prediction evaluation completed with HOTA/IDF1 zero. A VisDrone detector is not expected to detect the synthetic colored rectangles; these scores are not model-quality evidence.

Full configuration parameter counts (including the fixed YOLO DFL projection):

| Component | Parameters |
|---|---:|
| Falcon perception + new ReID | 7,589,870 |
| Object RSSM + neighbors | 448,464 |
| Camera dynamics | 40,716 |
| Association + gate + critic | 53,765 |
| **Total** | **8,132,815** |

Runtime metadata is in `validation_environment.json`. The existing system environment reports an old torchvision version and emits an Ultralytics compatibility warning; it was left untouched. Tests used isolated temporary Ultralytics/TrackEval installations. Use the clean Python/PyTorch/torchvision environment described in the README for actual training. No CUDA/AMP convergence test, full-resolution VRAM measurement, real-dataset training, or official benchmark score has been obtained.

# Per-track Life Agents: implemented v0.3 contract

This is a research implementation, not a claim of certified stability, a universal definition of artificial life, or confirmed paper novelty. Historical whole-scene hypothesis ecology is preserved in `METHOD_V1.md` and `legacy_tracking.py`.

## Causal state and policy

Each identity owns physical belief (h,z,b,v), working memory m, protected initial appearance anchor, existence probability, variance, missing age, hit count, lifetime, innovation EMA, phase and compute credit. All agents share neural weights. Current neighbor attention is shared object-centric RSSM inference.

The controller maps current hidden state and 13 available statistics to existence, lifecycle logits and two resource values. Candidate policy consumes association context, agent statistics and four physical-forecast summaries. It predicts match correctness and a residual on the base memory-write logit. Candidate critic additionally consumes gate, two forecast affinity summaries and action-type flags; it predicts future identity cost, memory amplification and cost uncertainty.

Tracker actions modify identity beliefs and memory, not physical objects:

p(x_next, observation_next, agent_next | x, agent, action)
= p_world(x_next | x, camera)
  p_observation(observation_next | x_next)
  p_tracker(agent_next | agent, observation_next, action).

An action may change a corrected belief and hence its later prediction; it does not control the true video. All current candidate actions read the same detached physical cache. No hypothetical appearance is written as an actual measurement.

## Budgeted imagination and real acquisition

Every frame has a global token budget and agents have replenished, capped credits. One selected-object forecast step or additional five-point ROI pooling counts as one token. Planning ranks requests by learned value and eligibility/credit, then caches one H-step forecast for the selected set before scoring write/freeze/wait actions. Candidate count does not multiply physical forecasts. ROI requests rank the learned recheck value; identical detection pools are cached and shared.

Current hidden states carry the full current neighbor context. Future neighbor interactions are restricted to selected agents, an explicit approximation. A camera prior is held constant for short rollouts. Forecast coordinates are normalized image coordinates/frame, not calibrated metric physics. Tokens bound these extra operations, not CNN cost, all-pairs association, peak memory, hardware milliseconds or FLOPs.

## Global arbitration and lifecycle

Candidate utilities combine class-gated association evidence, learned correctness, forecast affinity and critic risk plus uncertainty margin. Agent-specific null utilities account for wait/lifecycle and learned wait cost. Hungarian arbitration enforces one-to-one observation ownership. The selected action either writes observed appearance through a learned gate or freezes it.

Only actual assigned measurements correct physical state and reset uncertainty. Missing agents accumulate variance, retain memory and become dormant. Confirmation requires repeated real measurements. Actual later matches recover the same immutable ID. Learned termination requires low existence and a high terminal-policy probability after a grace interval; tentative and maximum dormant-age caps bound retained state. Current trusted observations protect against an erroneous early terminal decision. Birth candidates require detector evidence, and after policy warmup a learned existence check on birth features.

These guards remain configured constraints. IDs are not conserved at the expense of false-positive persistence: background negative candidates supervise birth/existence/termination and candidate quality, and lifecycle targets include known annotated track ends. No predicted-only boxes are exported as observed tracks.

## Training and causal labels

Joint optimization includes detector, ReID, RSSM/camera, association and LifeController. GT supplies detection/identity targets, GT-guided early observations and future loss/replay labels. Real detector proposals gradually replace a fraction of GT-jitter measurements. NMS/Hungarian/death are discrete; training keeps identity slots for supervision even when a deployed policy would terminate.

Sequence lifespans provide alive labels across internal missing annotations. Terminal labels are only known after the last annotation plus a grace interval if the sequence extends far enough. Sequence boundaries are censored. These are annotation-lifecycle labels, not ground truth of literal physical existence behind arbitrary occluders.

Independent and burst observation loss train dormant retention; working-memory corruption trains repair while preserving anchors and GT. Real background crops train negative tentative agents. Fresh birth features receive positive existence supervision from current correct object measurements, including after burn-in. Candidate correctness and write targets depend on assigned GT identity; stronger correct writes repair larger memory discrepancies.

Acquisition targets measure the change in correct-label association loss from real center features to a real five-point ROI. Planning requests learn a *potential* value-of-computation target: baseline replay cost minus the best tested cost for that identity. This is an oracle-improvement proxy on a bounded action set, not a measured guarantee of deployed planning benefit.

Paired intervention replay copies a tracker belief and uses the same cached observation stream for correct/swap assignments, write gates 0/.25/1 and explicit wait. Future matching uses the model, while GT only scores identity errors. Replay after the intervened instant retains the v1 fixed .15 continuation write rate. Therefore critic labels approximate local intervention value, rather than exact long-horizon value of the deployed controller.

The AL critic regresses replay cost/uncertainty and finite-time memory divergence, and learns within-identity risk ranking. A frozen-parameter critic path guides the memory proposal after warmup. Physical caches are detached so controller losses cannot alter a candidate's world to make its action appear better. Sample counters are incremented only at actual optimizer updates, and are checkpointed; they are warmup guards, not calibration certificates.

## Required experimental evidence

Before claiming benefit, measure identity recovery after occlusion/corruption, premature termination, ghost track duration, duplicate ownership, acquisition tokens, latency/VRAM, IDF1/HOTA/AssA/DetA and detector accuracy on held-out sequences. Compare matching training budgets and report multiple seeds. Ablate learned lifecycle, memory repair, acquisition and imagination independently, controlling inference as well as loss terms.

Limitations: protected anchors can themselves be poor, false-positive proposal sequence replay is incomplete, motion modes use mean-acceleration recurrent rollouts, termination is guarded supervised classification rather than differentiable life/death exploration, and future selected-agent simulation omits unselected future neighbors. No open-ended genetic evolution, biological metabolism, 3D physical simulation, formal safety certificate or proven AL emergence is asserted.

## v0.3 train/inference alignment

The tracker is trained on the candidate distribution it meets in deployment. Each frame's association softmax contains slot measurements (GT-jitter or matched detector proposals carrying the detector's class and confidence) plus up to `max_distractors` detector proposals that localize no visible object. Near-duplicates (IoU ≥ `proposal_iou` with a visible object) are excluded as neither right nor wrong. Free-running assignment can attach an agent to a distractor, which then corrupts its memory and state exactly as at inference. Confident distractors are linked by IoU into false tracks with negative existence/lifecycle/birth labels; ghosts are real false positives or background boxes drawn from the clip's object-size distribution.

Shared rules are single functions called by both training and the deployed tracker: confusable class groups (forbidden across groups, soft penalty inside), a hard position gate in units of the search scale, size-bounded missing uncertainty, per-frame velocity decay of unobserved agents, and group-agnostic NMS of proposals. Agent count statistics are saturating, and training adds random prior history at slot birth so that long-lived deployed agents stay in the trained input range. Critic contexts use the real anchor, age and ambiguity margin; replay continuation can follow the AL quality/null/gate policy (`on_policy_replay`; planning and critic are not rolled out inside replay).

Inference associates in two stages: confident detections by the learned utility, then IoU-only recovery of low-confidence boxes by agents seen within `low_conf_max_age` frames, without appearance writes. ROI pooling refines scores only. Births are suppressed when they duplicate a just-matched agent of the same group. Tentative observations are released (backfilled) at confirmation.

Remaining approximations: distractors are not replayed inside intervention probes; truncated BPTT across clips is not implemented (clips of 16 frames with stride ≤ 2 and bursts ≤ 10 bound the trained gap length); class groups are a fixed prior, not learned.

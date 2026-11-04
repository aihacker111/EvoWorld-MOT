# Historical v1 design (superseded by METHOD.md)

# EvoWorld-MOT: implemented method and experimental contract

This document describes code that exists in this repository. It does not assert a proven novelty claim or theoretical stability guarantee.

## State factorization

For identity i, maintain physical belief s_i=(h_i,z_i,b_i,v_i), immutable initial appearance anchor a_i, working appearance memory m_i, class and age. A camera GRU maintains g_t from compact shared-P2 observations. A hypothesis H contains a complete one-to-one identity assignment and its own state/memory; hypotheses share neural weights and current dense image features.

The causal factorization is:

p(x[t+1], y[t+1], H[t+1] | x[t], H[t], u[t])
= p_theta(x[t+1] | x[t], g[t])
  p_phi(y[t+1] | x[t+1])
  p_psi(H[t+1] | H[t], y[t+1], u[t]).

Here u is a **tracker action**, such as assigning a measurement or writing memory. Passive tracker actions cannot control physical objects or the camera. They affect beliefs, not the actual video. Correcting a belief can of course change its subsequent predicted distribution. The implementation does not synthesize an alternative video for each action.

## Physical prior and measurement posterior

ObjectRSSM applies shared local-neighbor attention and a GRU transition. Continuous latent prior and posterior are diagonal Gaussians; sampling is used during training, means during inference. New identities require an observation and receive a learned birth state.

For each acceleration mode k:

c^-_{i,t+1,k} = A_t(c_{i,t} + v_{i,t} dt + 0.5 a_{i,t,k} dt^2),
v^-_{i,t+1} = A_t.linear (v_{i,t} + mean_k(a_{i,t,k}) dt).

Size has a bounded learned log-rate and is mapped through the affine envelope. A learned diagonal uncertainty and categorical mixture weights define observation likelihood. A recurrent mean-acceleration transition is used for rollouts, rather than carrying exponentially many mode combinations. Forecast visibility and appearance are auxiliary targets. Thus the model is multimodal in one-step likelihood, but **does not maintain full multimodal trajectory distributions** over all future steps.

Units are normalized image coordinates and video frames. Camera motion is a 2D affine estimate from masked background optical flow/RANSAC, with a learned camera posterior as fallback and a learned prior for open-loop rollouts. Future camera priors use the current GRU state, held fixed over the short forecast horizon. There is no depth, calibrated metric speed, aerodynamic model, 3D collision model or conservation-law claim.

## Identity ecology

Each parent predicts its current object states once, then forms a class-gated score from learned residual affinity, working/anchor appearance similarity, geometric innovation, predicted uncertainty and visibility. Hungarian assignment includes unmatched columns.

Ambiguous matched edges generate alternative assignments by forbidding selected edges. Memory write alternatives inherit the same observation and freeze rather than overwrite memory. Each descendant updates its own posterior with actual assigned measurements, then adds sufficiently confident unmatched detections as births. A birth at the same frame/detection uses the same new ID across branches. Old tracks expire by max_age; births and matches are disjoint.

Memory update is:

m' = normalize((1-g)m + g e_observed),  g in [0,1].

No imagined embedding is written as a real observation. Anchors remain fixed. Selection uses decayed cumulative matching evidence, miss penalties and estimated future identity risk plus an uncertainty margin. Hypotheses persist across frames with bounded lineage metadata. Similar output/lineage signatures are deduplicated; this is an approximate bounded search, not exhaustive Bayesian filtering or globally optimal MHT. Online winner changes can still produce ID switches.

## Paired interventions

At a sampled clip anchor, copy tracker belief and replay the **same cached real appearance measurements** for a short horizon. Intervene once on a correct versus swapped assignment and memory write gates 0, .25 or 1. All subsequent associations use the same neural matcher and world model; GT labels do not choose future associations.

Targets are the number of erroneous/missing assignments of the probed identity and terminal memory divergence relative to a paired correct baseline, divided by the initial appearance perturbation with a numerical floor. The latter is an empirical finite-time amplification proxy, not a Jacobian spectral radius or a Lyapunov certificate. Replay uses at most two eligible targets and the nearest same-class distractor per target; it is a computationally bounded training distribution.

The effect critic predicts cost, gain and heteroscedastic cost uncertainty from **current** state/measurement/action context. It learns Gaussian cost regression, gain regression and pairwise ordering. A frozen-parameter critic path regularizes the learned memory gate after a minimum number of replay examples. Inference only activates the risk penalty after the same warmup guard. This guard does not certify calibration or safety; inspect held-out effects and tracking outcomes.

Baseline and intervention replay use a fixed .15 write rate after the intervened instant. Consequently replay measures a local intervention under that continuation policy, not exact long-horizon value of the deployed ecology. The approximate critic is one component of hypothesis scoring, not an oracle.

## Joint training

L = L_det + w_r L_ReID + w_a(L_assoc + .1 L_gate)
  + w_w(L_motion + .1 L_visibility + w_KL L_KL)
  + w_c L_camera + w_e L_effect + w_s L_stability.

L_det is the actual YOLO detection loss. ReID uses supervised contrastive positives across ordered frames. Association uses known training identities with an explicit null target. The latent objective uses free-nat KL and multi-step open-loop mixture likelihood; future appearance targets are detached. Camera pseudo-supervision is only applied when background estimation passes validity checks. Physical rollouts never consume future object/camera observations.

All learnable modules update in one optimizer run with separate pretrained/new-module learning rates. Ordered burn-in has no gradient by default. Teacher forcing decays, synthetic misses and box jitter introduce imperfect observations, and actual detector proposals replace GT-jitter observations on an increasing fraction of frames. Proposal→GT matching provides labels during training only. Unmatched false-positive proposals are not included in the current replay sampler; this remains a limitation to quantify on real detections.

NMS and Hungarian are discrete and detached. Detector convolutions receive detection, ReID, camera and temporal feature gradients, but not a fictitious differentiable path through NMS/Hungarian.

## Evaluation before any paper claim

Run real held-out sequences and the official evaluator for the chosen dataset protocol. Report IDF1, HOTA/AssA/DetA, ID switches, detector accuracy, runtime including perception and ecology, model size and peak GPU memory. Include multiple seeds and sequence bootstrap confidence intervals. Keep detector initialization, data splits, training budget and preprocessing fixed.

Useful experiments include population budget 1 versus 8, disabling intervention penalties, disabling the write-policy regularizer, proposal curriculum on/off, and varying temporal horizon. An ablation that disables a loss does not remove that module from inference; implement and document the corresponding inference baseline when making causal claims about a component.

Benchmark quality, world-model calibration, critic calibration, long-occlusion recovery, crowded tiny-object scenes and camera zoom are **not established by synthetic smoke tests**. No training run on the actual target dataset or CUDA convergence experiment is bundled.

"""Joint detector, appearance, latent dynamics, association and intervention training."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from evoworld.geometry import sample_features, warp_boxes, pair_iou
from evoworld.models.association import assign, blend_memory
from evoworld.interventions import intervention_loss
from evoworld.models.life import AgentState, uncertainty_forecast
from evoworld.al_learning import (
    censored_labels,
    observation_masks,
    corrupt_memory,
    frame_supervision,
    ghost_supervision,
    memory_supervision,
    effect_supervision,
)


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def motion_nll(info, target):
    sigma = info["log_sigma"].exp()
    residual = (target[..., None, :] - info["mode_boxes"]) / sigma[..., None, :]
    log_prob = -0.5 * residual.square().sum(-1) - info["log_sigma"].sum(-1)[..., None]
    log_prob = log_prob + info["mode_logits"].log_softmax(-1)
    return -torch.logsumexp(log_prob.float(), -1) / 4


def contrastive(features, ids, temperature=0.1):
    if len(ids) < 2:
        return features.sum() * 0
    similarity = features.float() @ features.float().T / temperature
    diagonal = torch.eye(len(ids), dtype=torch.bool, device=ids.device)
    positive = (ids[:, None] == ids[None]) & ~diagonal
    valid = positive.any(-1)
    if not valid.any():
        return features.sum() * 0
    denominator = torch.logsumexp(similarity.masked_fill(diagonal, -1e4), -1)
    numerator = torch.logsumexp(similarity.masked_fill(~positive, -1e4), -1)
    return (denominator - numerator)[valid].mean()


def training_objective(model, batch, epoch=0, step=0):
    c = model.config
    images, boxes, present = batch["images"], batch["boxes"], batch["present"]
    B, T, N, _ = boxes.shape
    use_al = c.tracker_mode == "al"
    gt_alive, life_known = censored_labels(batch)
    zero = images.new_zeros(())
    values = {
        key: zero
        for key in (
            "detector",
            "world",
            "visibility",
            "kl",
            "camera",
            "association",
            "gate",
            "effect",
            "stability",
            "al_lifecycle",
            "al_memory",
            "al_effect",
            "al_actor",
        )
    }
    features, cameras, appearances, dense_maps = [], [], [], []
    measurement_present, measurement_confidence = [], []
    for t in range(T):
        enabled = not (c.burn_in_no_grad and t < c.burn_in)
        with torch.set_grad_enabled(torch.is_grad_enabled() and enabled):
            frame = model.perception(images[:, t])
            detection = model.perception.detection_loss(
                frame.raw, images[:, t], boxes[:, t], batch["classes"], present[:, t]
            )
            dense_features = sample_features(frame.appearance, boxes[:, t])
        if enabled:
            values["detector"] = values["detector"] + detection / (T - c.burn_in)
        features.append(dense_features)
        dense_maps.append(frame.appearance)
        cameras.append(frame.camera)
        # Measurements are jittered crops, supervision is centered on true visible objects.
        jitter = torch.randn_like(boxes[:, t]) * c.box_noise * boxes[:, t, :, 2:].mean(-1, keepdim=True)
        measured = boxes[:, t] + jitter
        measured[..., :2] = measured[..., :2].clamp(0, 1)
        measured[..., 2:] = measured[..., 2:].clamp_min(1e-4)
        available = present[:, t].clone()
        confidence = images.new_ones(B, N)
        proposal_probability = c.proposal_probability * min(1.0, epoch / max(1, c.warmup_epochs))
        if c.backend == "falcon" and proposal_probability > 0:
            proposals = model.perception.training_detections(frame.raw)
            for b, proposal in enumerate(proposals):
                if torch.rand((), device=images.device) >= proposal_probability:
                    continue
                overlap = pair_iou(boxes[b, t], proposal["boxes"])
                valid = (batch["classes"][b, :, None] == proposal["classes"][None]) & present[b, t, :, None]
                overlap = overlap.masked_fill(~valid, -1.0)
                available[b].zero_()
                confidence[b].zero_()
                for i, j in assign(overlap, c.proposal_iou):
                    measured[b, i] = proposal["boxes"][j]
                    available[b, i] = True
                    confidence[b, i] = proposal["scores"][j]
        appearances.append((measured, sample_features(frame.appearance, measured)))
        measurement_present.append(available)
        measurement_confidence.append(confidence)
    observed_sequence = observation_masks(torch.stack(measurement_present, 1), c)
    appearance_tensor = torch.stack(features, 1)
    identity = batch["ids"][:, None].expand(B, T, N)
    mask = present & (identity >= 0)
    reid = contrastive(appearance_tensor[mask], identity[mask])
    state = model.world.initial(images, B, N)
    memory = images.new_zeros(B, N, c.embedding_dim)
    anchor = memory.clone()
    camera_state = None
    agents = [AgentState.initial(images, N) for _ in range(B)] if use_al else []
    life_snapshots, camera_snapshots = {}, {}
    policy_examples, critic_examples = 0, 0
    forcing = c.teacher_forcing_start + (c.teacher_forcing_end - c.teacher_forcing_start) * min(
        epoch / max(1, c.epochs - 1), 1.0
    )
    anchors = []
    observed_motions = []
    rollout_terms = 0
    for t in range(T):
        if use_al and t >= c.burn_in:
            memory = corrupt_memory(memory, state.valid, c.al_memory_corruption)
        previous_camera = cameras[max(t - 1, 0)]
        old_camera_state = camera_state
        camera_state, observed_camera = model.camera.observe(cameras[t], previous_camera, camera_state)
        valid_camera = batch["camera_valid"][:, t]
        if t >= c.burn_in:
            error = (observed_camera - batch["affine"][:, t]).square().mean((-1, -2))
            values["camera"] = values["camera"] + masked_mean(error, valid_camera) / (T - c.burn_in)
            if old_camera_state is not None:
                prior_error = (
                    (model.camera.imagine(old_camera_state) - batch["affine"][:, t]).square().mean((-1, -2))
                )
                values["camera"] = values["camera"] + masked_mean(prior_error, valid_camera) / (T - c.burn_in)
        motion = torch.where(valid_camera[:, None, None], batch["affine"][:, t], observed_camera)
        observed_motions.append(motion)
        camera_snapshots[t] = camera_state.detach()
        previous_box = warp_boxes(state.box, motion)
        predicted, info = model.world.predict(state, motion, batch["delta"][:, t], sample=model.training)
        observed = observed_sequence[:, t]
        measurement_boxes, measurement_features = appearances[t]
        corrected_boxes, corrected_features = measurement_boxes.clone(), measurement_features.clone()
        corrected_observed = observed.clone()
        corrected_confidence = measurement_confidence[t].clone()
        correct_identity = observed.clone()
        new_memory, new_anchor = [], []
        for b in range(B):
            local_forecast = {key: value[b : b + 1] for key, value in info.items()}
            if use_al:
                current_agent = agents[b].predict(local_forecast, batch["delta"][b : b + 1, t], c)
                local_forecast = uncertainty_forecast(local_forecast, current_agent)
            scores = model.association.scores(
                predicted.select(slice(b, b + 1)),
                memory[b],
                anchor[b],
                measurement_boxes[b],
                measurement_features[b],
                batch["classes"][b],
                batch["classes"][b],
                local_forecast,
            )
            scores = scores.masked_fill(~observed[b][None], -1e4)
            learn_rows = state.valid[b]
            if use_al:
                life_loss, life_stats, summary, policy, count, _ = frame_supervision(
                    model,
                    predicted.select(slice(b, b + 1)),
                    local_forecast,
                    current_agent,
                    scores,
                    memory[b],
                    anchor[b],
                    measurement_boxes[b],
                    measurement_features[b],
                    dense_maps[t][b : b + 1],
                    batch["classes"][b],
                    observed[b],
                    gt_alive[b, t],
                    life_known[b, t],
                    model.camera.imagine(camera_state[b : b + 1]),
                    batch["delta"][b : b + 1, t],
                )
                if t >= c.burn_in:
                    negative = ghost_supervision(model, dense_maps[t][b : b + 1], boxes[b, t], present[b, t])
                    values["al_lifecycle"] = values["al_lifecycle"] + (life_loss + 0.2 * negative) / (
                        B * (T - c.burn_in)
                    )
                    policy_examples += count + 1
            if t >= c.burn_in and learn_rows.any():
                logits = torch.cat((scores, scores.new_full((N, 1), c.match_threshold)), -1)
                target = torch.arange(N, device=images.device)
                target = torch.where(observed[b], target, N)
                values["association"] = values["association"] + F.cross_entropy(
                    logits[learn_rows] / c.likelihood_temperature, target[learn_rows]
                ) / (B * (T - c.burn_in))
            teacher = torch.rand((), device=images.device) < forcing
            if not teacher and learn_rows.any():
                pairs = assign(scores.masked_fill(~learn_rows[:, None], -1e4), c.match_threshold)
                corrected_observed[b] = observed[b] & ~learn_rows
                correct_identity[b] = corrected_observed[b].clone()
                for i, j in pairs:
                    corrected_boxes[b, i] = measurement_boxes[b, j]
                    corrected_features[b, i] = measurement_features[b, j]
                    corrected_observed[b, i] = True
                    correct_identity[b, i] = i == j
                    corrected_confidence[b, i] = measurement_confidence[t][b, j]
            contexts = model.association.context(
                predicted.h[b],
                memory[b],
                anchor[b],
                corrected_features[b],
                predicted.box[b],
                corrected_boxes[b],
                corrected_confidence[b],
                current_agent.age if use_al else images.new_zeros(N),
                life_stats[:, 9] * 5 if use_al else images.new_zeros(N),
            )
            gate = model.association.gate(contexts)
            prior_gate = gate
            if use_al:
                gate, maintenance_loss = memory_supervision(
                    model,
                    contexts,
                    life_stats,
                    summary,
                    gate,
                    memory[b],
                    corrected_features[b],
                    corrected_observed[b],
                    correct_identity[b],
                )
                if t >= c.burn_in:
                    values["al_memory"] = values["al_memory"] + maintenance_loss / (B * (T - c.burn_in))
                innovation = (corrected_boxes[b, :, :2] - predicted.box[b, :, :2]).norm(dim=-1)
                innovation = innovation / predicted.box[b, :, 2:].mean(-1).clamp_min(0.02)
                agents[b] = current_agent.observe(
                    corrected_observed[b],
                    corrected_confidence[b],
                    corrected_boxes[b],
                    innovation,
                    policy["existence"].sigmoid(),
                    True,
                    c,
                )
            if t >= c.burn_in:
                # Low-rate writes on consistent observations; counterfactual critic refines this policy.
                target_gate = correct_identity[b].float() * 0.2
                gate_loss = F.binary_cross_entropy_with_logits(
                    torch.logit(prior_gate.float().clamp(1e-5, 1 - 1e-5)),
                    target_gate.float(),
                    reduction="none",
                )
                values["gate"] = values["gate"] + masked_mean(gate_loss, corrected_observed[b]) / (
                    B * (T - c.burn_in)
                )
            born = corrected_observed[b] & ~state.valid[b]
            updated = blend_memory(memory[b], corrected_features[b], gate)
            updated = torch.where(born[:, None], corrected_features[b], updated)
            new_memory.append(torch.where(corrected_observed[b, :, None], updated, memory[b]))
            new_anchor.append(torch.where(born[:, None], corrected_features[b], anchor[b]))
        memory, anchor = torch.stack(new_memory), torch.stack(new_anchor)
        state, kl = model.world.correct(
            predicted,
            corrected_boxes,
            corrected_features,
            corrected_observed,
            batch["delta"][:, t],
            previous_box,
            sample=model.training,
        )
        if use_al and t >= c.burn_in:
            for b in range(B):
                mask_birth = corrected_observed[b] & correct_identity[b] & gt_alive[b, t]
                if mask_birth.any():
                    encoded_birth = model.world.obs(
                        torch.cat(
                            (
                                corrected_features[b, mask_birth],
                                corrected_boxes[b, mask_birth],
                                images.new_ones(int(mask_birth.sum()), 1),
                            ),
                            -1,
                        )
                    )
                    birth_hidden = model.world.birth(encoded_birth)
                    birth_policy = model.al.agents(
                        birth_hidden,
                        model.al.birth_statistics(corrected_confidence[b, mask_birth], c),
                    )
                    birth_loss = F.binary_cross_entropy_with_logits(
                        birth_policy["existence"], torch.ones_like(birth_policy["existence"])
                    )
                    values["al_lifecycle"] = values["al_lifecycle"] + birth_loss / (B * (T - c.burn_in))
                    policy_examples += int(mask_birth.sum())
        if t < c.burn_in:
            state, memory, anchor = state.detached(), memory.detach(), anchor.detach()
            camera_state = camera_state.detach()
            continue
        values["kl"] = values["kl"] + masked_mean(kl.clamp_min(c.free_nats), corrected_observed) / (
            T - c.burn_in
        )
        anchors.append((t, state, memory, camera_state))
        if use_al:
            life_snapshots[t] = [agent.detached() for agent in agents]
        # Open-loop priors never consume future object observations or future camera evidence.
        imagined = state
        for k in range(1, min(c.horizon, T - 1 - t) + 1):
            imagined_camera = model.camera.imagine(camera_state)
            imagined, forecast = model.world.predict(
                imagined, imagined_camera, batch["delta"][:, t + k], sample=model.training
            )
            known = state.valid & present[:, t + k]
            values["world"] = values["world"] + masked_mean(motion_nll(forecast, boxes[:, t + k]), known)
            values["world"] = values["world"] + 0.2 * masked_mean(
                1 - (forecast["feature"] * features[t + k].detach()).sum(-1), known
            )
            values["visibility"] = values["visibility"] + masked_mean(
                F.binary_cross_entropy_with_logits(
                    forecast["visibility"], present[:, t + k].float(), reduction="none"
                ),
                state.valid,
            )
            rollout_terms += 1
    values["world"] = values["world"] / max(1, rollout_terms)
    values["visibility"] = values["visibility"] / max(1, rollout_terms)
    probes = 0
    if step % c.probe_every == 0 and anchors:
        t, saved_state, saved_memory, _ = anchors[0]
        for b in range(B):
            observations = [
                (
                    appearances[s][0][b],
                    appearances[s][1][b],
                    observed_sequence[b, s],
                    observed_motions[s][b : b + 1],
                    batch["delta"][b : b + 1, s],
                    measurement_confidence[s][b],
                )
                for s in range(t + 1, min(T, t + 1 + c.probe_horizon))
            ]
            result = intervention_loss(
                model,
                saved_state.select(slice(b, b + 1)),
                saved_memory[b],
                batch["classes"][b],
                observations,
                collect=use_al,
            )
            effect, stability, count = result[:3]
            if use_al:
                learned_effect, actor, examples = effect_supervision(
                    model,
                    result[3],
                    life_snapshots[t][b],
                    saved_memory[b].detach(),
                    batch["classes"][b],
                    model.camera.imagine(camera_snapshots[min(t + 1, T - 1)][b : b + 1]),
                    observations,
                )
                values["al_effect"] = values["al_effect"] + learned_effect / B
                values["al_actor"] = values["al_actor"] + actor / B
                critic_examples += examples
            values["effect"] = values["effect"] + effect / B
            values["stability"] = values["stability"] + stability / B
            probes += count
    loss = (
        values["detector"]
        + c.reid_weight * reid
        + c.association_weight * (values["association"] + 0.1 * values["gate"])
        + c.world_weight * (values["world"] + 0.1 * values["visibility"] + c.kl_weight * values["kl"])
        + c.camera_weight * values["camera"]
        + c.effect_weight * values["effect"]
        + c.stability_weight * values["stability"]
        + c.al_weight
        * (values["al_lifecycle"] + values["al_memory"] + values["al_effect"] + 0.1 * values["al_actor"])
    )
    metrics = {key: float(value.detach()) for key, value in values.items()}
    metrics.update(
        loss=float(loss.detach()),
        reid=float(reid.detach()),
        teacher_forcing=forcing,
        probes=probes,
        al_policy_examples=policy_examples,
        al_critic_examples=critic_examples,
    )
    return loss, metrics

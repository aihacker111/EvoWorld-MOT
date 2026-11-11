"""Joint detector, appearance, latent dynamics, association and intervention training.

The tracker is trained on the candidate distribution it meets at inference:
slot measurements (GT-jitter or matched detector proposals, with detector
classes) plus detector false positives as distractor columns.  Optionally,
confident false positives become their own tracks with negative lifecycle
labels.  Shared rules (class groups, position gate, missing-velocity damping,
bounded uncertainty) are the same functions the deployed tracker calls.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from evoworld.geometry import sample_features, warp_boxes, pair_iou
from evoworld.classes import same_group
from evoworld.models.association import assign, blend_memory
from evoworld.models.world import damp_missing
from evoworld.interventions import intervention_loss
from evoworld.models.life import AgentState, uncertainty_forecast
from evoworld.al_learning import (
    censored_labels,
    candidate_targets,
    observation_masks,
    corrupt_memory,
    frame_supervision,
    ghost_boxes,
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


def proposal_schedule(config, epoch):
    """Warmup ramp to proposal_probability, then optional linear ramp to the end value."""
    warmup = max(1, config.warmup_epochs)
    probability = config.proposal_probability * min(1.0, epoch / warmup)
    end = config.proposal_probability_end
    if end is not None and epoch >= warmup:
        progress = min(1.0, (epoch - warmup) / max(1, config.epochs - 1 - warmup))
        probability = config.proposal_probability + (end - config.proposal_probability) * progress
    return probability


def finite_scores(scores):
    return torch.where(torch.isfinite(scores), scores, scores.new_full((), -1e4))


def perception_grad(config, t):
    if config.burn_in_no_grad and t < config.burn_in:
        return False
    return (t - (config.burn_in if config.burn_in_no_grad else 0)) % config.perception_grad_every == 0


def link_false_tracks(distractors, config, limit):
    """Greedy IoU tracklets over confident distractors of one clip.

    Returns up to `limit` tracklets as dict(frames={t: distractor index}, cls=class),
    highest total confidence first.
    """
    tracklets = []
    for t, (boxes, _, confidence, classes) in enumerate(distractors):
        candidates = (confidence >= config.birth_threshold).nonzero().flatten().tolist()
        active = [k for k, track in enumerate(tracklets) if t - track["last"] <= 2]
        used = set()
        if active and candidates:
            last = torch.stack([tracklets[k]["box"] for k in active])
            rows = torch.tensor(candidates, device=boxes.device)
            overlap = pair_iou(last, boxes[rows])
            classes_last = torch.stack([tracklets[k]["cls"] for k in active])
            overlap = overlap.masked_fill(~same_group(classes_last, classes[rows], config), -1.0)
            for a, b in assign(overlap, 0.3):
                track, j = tracklets[active[a]], candidates[b]
                track["frames"][t] = j
                track.update(box=boxes[j], cls=classes[j], last=t)
                track["score"] += float(confidence[j])
                used.add(j)
        for j in candidates:
            if j not in used:
                tracklets.append(
                    dict(frames={t: j}, box=boxes[j], cls=classes[j], last=t, score=float(confidence[j]))
                )
    tracklets.sort(key=lambda track: -track["score"])
    return [dict(frames=track["frames"], cls=track["cls"]) for track in tracklets[:limit]]


def training_objective(model, batch, epoch=0, step=0):
    c = model.config
    images, gt_boxes, gt_present = batch["images"], batch["boxes"], batch["present"]
    B, T, N0, _ = gt_boxes.shape
    use_al = c.tracker_mode == "al"
    legacy = not use_al or c.legacy_effect_in_al
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
    proposal_probability = proposal_schedule(c, epoch)
    # Falcon exposes decoded proposals from the training head; the tiny test backend does not.
    can_decode = callable(getattr(model.perception, "training_detections", None))
    decode = can_decode and (proposal_probability > 0 or c.fp_distractors or c.fp_tracks)
    grad_frames = [t for t in range(T) if perception_grad(c, t)]
    features, cameras, dense_maps = [], [], []
    slot_boxes, slot_features, slot_present, slot_confidence, slot_classes = [], [], [], [], []
    distractors = [[None] * T for _ in range(B)]
    distractor_count = 0
    for t in range(T):
        enabled = perception_grad(c, t)
        with torch.set_grad_enabled(torch.is_grad_enabled() and enabled):
            frame = model.perception(images[:, t])
            if enabled:
                detection = model.perception.detection_loss(
                    frame.raw, images[:, t], gt_boxes[:, t], batch["classes"], gt_present[:, t]
                )
                values["detector"] = values["detector"] + detection / max(1, len(grad_frames))
            dense_features = sample_features(frame.appearance, gt_boxes[:, t])
        features.append(dense_features)
        dense_maps.append(frame.appearance)
        cameras.append(frame.camera)
        # Measurements are jittered crops, supervision is centered on true visible objects.
        jitter = torch.randn_like(gt_boxes[:, t]) * c.box_noise * gt_boxes[:, t, :, 2:].mean(-1, keepdim=True)
        measured = gt_boxes[:, t] + jitter
        measured[..., :2] = measured[..., :2].clamp(0, 1)
        measured[..., 2:] = measured[..., 2:].clamp_min(1e-4)
        available = gt_present[:, t].clone()
        # Synthetic detector confidence for GT-jitter measurements: confidence is an association
        # input, so a constant 1.0 would become a trivial "real object" shortcut.
        confidence = c.conf_threshold + (1 - c.conf_threshold) * torch.rand(B, N0, device=images.device)
        measured_classes = batch["classes"].clone()
        proposals = model.perception.training_detections(frame.raw) if decode else [None] * B
        for b, proposal in enumerate(proposals):
            empty = images.new_zeros(0, 4)
            distractors[b][t] = (
                empty,
                images.new_zeros(0, c.embedding_dim),
                images.new_zeros(0),
                torch.zeros(0, dtype=torch.long, device=images.device),
            )
            if proposal is None:
                continue
            overlap = pair_iou(gt_boxes[b, t], proposal["boxes"])
            valid = same_group(batch["classes"][b], proposal["classes"], c) & gt_present[b, t, :, None]
            overlap = overlap.masked_fill(~valid, -1.0)
            pairs = assign(overlap, c.proposal_iou)
            if torch.rand((), device=images.device) < proposal_probability:
                available[b].zero_()
                confidence[b].zero_()
                for i, j in pairs:
                    measured[b, i] = proposal["boxes"][j]
                    available[b, i] = True
                    confidence[b, i] = proposal["scores"][j]
                    measured_classes[b, i] = proposal["classes"][j]
            if not (c.fp_distractors or c.fp_tracks) or not len(proposal["boxes"]):
                continue
            # Distractors: proposals that are not a localization of any visible object.
            # Near-duplicates (IoU >= proposal_iou) are neither right nor wrong and are excluded.
            used = torch.zeros(len(proposal["boxes"]), dtype=torch.bool, device=images.device)
            used[[j for _, j in pairs]] = True
            if gt_present[b, t].any():
                best = pair_iou(proposal["boxes"], gt_boxes[b, t, gt_present[b, t]]).max(-1).values
                used |= best >= c.proposal_iou
            index = (~used).nonzero().flatten()
            index = index[proposal["scores"][index].argsort(descending=True)[: c.max_distractors]]
            boxes_d = proposal["boxes"][index]
            distractors[b][t] = (
                boxes_d,
                sample_features(frame.appearance[b : b + 1], boxes_d[None])[0],
                proposal["scores"][index],
                proposal["classes"][index],
            )
            distractor_count += len(index)
        slot_boxes.append(measured)
        slot_features.append(sample_features(frame.appearance, measured))
        slot_present.append(available)
        slot_confidence.append(confidence)
        slot_classes.append(measured_classes)
    # ----- Detector false positives that would become tracks: extra identity slots. -----
    tracklets = [[] for _ in range(B)]
    if use_al and c.fp_tracks and decode:
        for b in range(B):
            tracklets[b] = link_false_tracks(distractors[b], c, c.max_fp_tracks)
    K = max((len(x) for x in tracklets), default=0)
    N = N0 + K
    track_classes = torch.cat(
        (batch["classes"], torch.zeros(B, K, dtype=torch.long, device=images.device)), 1
    )
    false_track = torch.zeros(B, N, dtype=torch.bool, device=images.device)
    real = torch.zeros(B, N, dtype=torch.bool, device=images.device)
    real[:, :N0] = True
    present = torch.cat((gt_present, gt_present.new_zeros(B, T, K)), 2)
    boxes = torch.cat((gt_boxes, gt_boxes.new_zeros(B, T, K, 4)), 2)
    alive = torch.cat((gt_alive, gt_alive.new_zeros(B, T, K)), 2)
    known = torch.cat((life_known, life_known.new_zeros(B, T, K)), 2)

    def pad(tensor, value=0):
        shape = list(tensor.shape)
        shape[1] = K
        return torch.cat((tensor, tensor.new_full(shape, value)), 1)

    slot_boxes = [pad(x) for x in slot_boxes]
    slot_features = [pad(x) for x in slot_features]
    slot_present = [pad(x, False) for x in slot_present]
    slot_confidence = [pad(x) for x in slot_confidence]
    slot_classes = [pad(x) for x in slot_classes]
    features = [pad(x) for x in features]
    for b in range(B):
        removed = [set() for _ in range(T)]
        for k, track in enumerate(tracklets[b]):
            slot = N0 + k
            track_classes[b, slot] = track["cls"]
            false_track[b, slot] = True
            known[b, :, slot] = True
            for t, j in track["frames"].items():
                boxes_d, features_d, confidence_d, classes_d = distractors[b][t]
                slot_boxes[t][b, slot] = boxes_d[j]
                slot_features[t][b, slot] = features_d[j]
                slot_present[t][b, slot] = True
                slot_confidence[t][b, slot] = confidence_d[j]
                slot_classes[t][b, slot] = classes_d[j]
                present[b, t, slot] = True
                boxes[b, t, slot] = boxes_d[j]
                removed[t].add(j)
        # A proposal that became a false-track measurement is no longer a distractor column.
        for t in range(T):
            if removed[t]:
                keep = torch.ones(len(distractors[b][t][0]), dtype=torch.bool, device=images.device)
                keep[list(removed[t])] = False
                distractors[b][t] = tuple(x[keep] for x in distractors[b][t])
    observed_sequence = observation_masks(torch.stack(slot_present, 1), c)
    appearance_tensor = torch.stack(features, 1)[:, :, :N0]
    identity = batch["ids"][:, None].expand(B, T, N0)
    mask = gt_present & (identity >= 0)
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
    rollout_weight = 0.0
    learning_frames = T - c.burn_in
    grad_enabled = torch.is_grad_enabled()
    for t in range(T):
        learning = t >= c.burn_in
        with torch.set_grad_enabled(grad_enabled and (learning or not c.burn_in_no_grad)):
            if use_al and learning:
                memory = corrupt_memory(memory, state.valid, c.al_memory_corruption)
            previous_camera = cameras[max(t - 1, 0)]
            old_camera_state = camera_state
            camera_state, observed_camera = model.camera.observe(cameras[t], previous_camera, camera_state)
            valid_camera = batch["camera_valid"][:, t]
            if learning:
                error = (observed_camera - batch["affine"][:, t]).square().mean((-1, -2))
                values["camera"] = values["camera"] + masked_mean(error, valid_camera) / learning_frames
                if old_camera_state is not None:
                    prior_error = (
                        (model.camera.imagine(old_camera_state) - batch["affine"][:, t]).square().mean((-1, -2))
                    )
                    values["camera"] = values["camera"] + masked_mean(prior_error, valid_camera) / learning_frames
            motion = torch.where(valid_camera[:, None, None], batch["affine"][:, t], observed_camera)
            observed_motions.append(motion)
            camera_snapshots[t] = camera_state.detach()
            previous_box = warp_boxes(state.box, motion)
            predicted, info = model.world.predict(state, motion, batch["delta"][:, t], sample=model.training)
            observed = observed_sequence[:, t]
            corrected = []
            new_memory, new_anchor = [], []
            for b in range(B):
                d_boxes, d_features, d_confidence, d_classes = distractors[b][t]
                F_ = len(d_boxes)
                C = N + F_
                candidate_boxes = torch.cat((slot_boxes[t][b], d_boxes))
                candidate_features = torch.cat((slot_features[t][b], d_features.to(slot_features[t].dtype)))
                candidate_confidence = torch.cat((slot_confidence[t][b], d_confidence.to(slot_confidence[t].dtype)))
                candidate_classes = torch.cat((slot_classes[t][b], d_classes))
                candidate_available = torch.cat((observed[b], observed.new_ones(F_)))
                # Each clip owns its correction buffers. Views into a shared batch
                # must not be mutated after another clip saves them for backward.
                local_boxes = slot_boxes[t][b].clone()
                local_features = slot_features[t][b].clone()
                local_observed = observed[b].clone()
                local_confidence = slot_confidence[t][b].clone()
                local_identity = observed[b].clone()
                local_forecast = {key: value[b : b + 1] for key, value in info.items()}
                if use_al:
                    current_agent = agents[b].predict(
                        local_forecast, batch["delta"][b : b + 1, t], c, predicted.box[b]
                    )
                    local_forecast = uncertainty_forecast(local_forecast, current_agent)
                scores = model.association.scores(
                    predicted.select(slice(b, b + 1)),
                    memory[b],
                    anchor[b],
                    candidate_boxes,
                    candidate_features,
                    track_classes[b],
                    candidate_classes,
                    local_forecast,
                    candidate_confidence,
                )
                scores = finite_scores(scores).masked_fill(~candidate_available[None], -1e4)
                learn_rows = state.valid[b]
                target = candidate_targets(observed[b], C)
                # A target excluded by the class group or position gate cannot be learned.
                feasible = (target == C) | (
                    scores.gather(1, target.clamp_max(C - 1)[:, None])[:, 0] > -1e3
                )
                if use_al:
                    life_loss, life_stats, summary, policy, count, _ = frame_supervision(
                        model,
                        predicted.select(slice(b, b + 1)),
                        local_forecast,
                        current_agent,
                        scores,
                        memory[b],
                        anchor[b],
                        (
                            candidate_boxes,
                            candidate_features,
                            candidate_confidence,
                            candidate_classes,
                            candidate_available,
                        ),
                        dense_maps[t][b : b + 1],
                        track_classes[b],
                        observed[b],
                        alive[b, t],
                        known[b, t],
                        model.camera.imagine(camera_state[b : b + 1]),
                        batch["delta"][b : b + 1, t],
                        false_track[b],
                    )
                    if learning:
                        ghost_box = d_boxes[d_confidence.argsort(descending=True)[: c.ghost_per_frame]]
                        ghost_score = d_confidence[d_confidence.argsort(descending=True)[: c.ghost_per_frame]]
                        extra = c.ghost_per_frame - len(ghost_box)
                        if extra > 0:
                            random_box = ghost_boxes(boxes[b, t, :N0], gt_present[b, t], extra, d_boxes)
                            ghost_box = torch.cat((ghost_box, random_box))
                            ghost_score = torch.cat(
                                (ghost_score, 0.5 + 0.45 * torch.rand(len(random_box), device=images.device))
                            )
                        negative = ghost_supervision(model, dense_maps[t][b : b + 1], ghost_box, ghost_score)
                        values["al_lifecycle"] = values["al_lifecycle"] + (life_loss + 0.2 * negative) / (
                            B * learning_frames
                        )
                        policy_examples += count + 1
                rows = learn_rows & feasible
                if learning and rows.any():
                    logits = torch.cat((scores, scores.new_full((N, 1), c.match_threshold)), -1)
                    values["association"] = values["association"] + F.cross_entropy(
                        logits[rows] / c.likelihood_temperature, target[rows]
                    ) / (B * learning_frames)
                teacher = torch.rand((), device=images.device) < forcing
                if not teacher and learn_rows.any():
                    # Free-running assignment over the full candidate set: an agent can drift onto
                    # a false positive, exactly as in deployment, and must learn to recover.
                    pairs = assign(scores.masked_fill(~learn_rows[:, None], -1e4), c.match_threshold)
                    local_observed = observed[b] & ~learn_rows
                    local_identity = local_observed.clone()
                    for i, j in pairs:
                        local_boxes[i] = candidate_boxes[j]
                        local_features[i] = candidate_features[j]
                        local_observed[i] = True
                        local_identity[i] = i == j
                        local_confidence[i] = candidate_confidence[j]
                contexts = model.association.context(
                    predicted.h[b],
                    memory[b],
                    anchor[b],
                    local_features,
                    predicted.box[b],
                    local_boxes,
                    local_confidence,
                    current_agent.age if use_al else images.new_zeros(N),
                    life_stats[:, 9] * 5 if use_al else images.new_zeros(N),
                )
                gate = model.association.gate(contexts)
                prior_gate = gate
                if use_al:
                    negatives = None
                    rows_with_fp = learn_rows.nonzero().flatten()
                    if F_ and len(rows_with_fp):
                        # Hardest false-positive column per agent: quality and write must be low.
                        hardest = scores[rows_with_fp, N:].argmax(-1)
                        reachable = scores[rows_with_fp, N + hardest] > -1e3
                        rows_with_fp, hardest = rows_with_fp[reachable], hardest[reachable]
                        if len(rows_with_fp):
                            negative_context = model.association.context(
                                predicted.h[b, rows_with_fp],
                                memory[b, rows_with_fp],
                                anchor[b, rows_with_fp],
                                d_features[hardest].to(memory.dtype),
                                predicted.box[b, rows_with_fp],
                                d_boxes[hardest],
                                d_confidence[hardest].to(memory.dtype),
                                current_agent.age[rows_with_fp],
                                life_stats[rows_with_fp, 9] * 5,
                            )
                            negatives = (
                                negative_context,
                                life_stats[rows_with_fp],
                                summary[rows_with_fp],
                                model.association.gate(negative_context),
                            )
                    gate, maintenance_loss = memory_supervision(
                        model,
                        contexts,
                        life_stats,
                        summary,
                        gate,
                        memory[b],
                        local_features,
                        local_observed,
                        local_identity,
                        negatives,
                    )
                    if learning:
                        values["al_memory"] = values["al_memory"] + maintenance_loss / (B * learning_frames)
                    innovation = (local_boxes[:, :2] - predicted.box[b, :, :2]).norm(dim=-1)
                    innovation = innovation / predicted.box[b, :, 2:].mean(-1).clamp_min(0.02)
                    agents[b] = current_agent.observe(
                        local_observed,
                        local_confidence,
                        local_boxes,
                        innovation,
                        policy["existence"].sigmoid(),
                        True,
                        c,
                    ).with_history(local_observed & ~state.valid[b], c)
                if learning and legacy:
                    # v1 low-rate write target; disabled in AL runs where the AL gate owns writes.
                    target_gate = local_identity.float() * 0.2
                    gate_loss = F.binary_cross_entropy_with_logits(
                        torch.logit(prior_gate.float().clamp(1e-5, 1 - 1e-5)),
                        target_gate.float(),
                        reduction="none",
                    )
                    values["gate"] = values["gate"] + masked_mean(gate_loss, local_observed) / (
                        B * learning_frames
                    )
                born = local_observed & ~state.valid[b]
                updated = blend_memory(memory[b], local_features, gate)
                updated = torch.where(born[:, None], local_features, updated)
                new_memory.append(torch.where(local_observed[:, None], updated, memory[b]))
                new_anchor.append(torch.where(born[:, None], local_features, anchor[b]))
                corrected.append((local_boxes, local_features, local_observed, local_confidence, local_identity))
            corrected_boxes, corrected_features, corrected_observed, corrected_confidence, correct_identity = (
                torch.stack(rows) for rows in zip(*corrected)
            )
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
            state = damp_missing(state, corrected_observed, batch["delta"][:, t], c.dormant_velocity_decay)
            if use_al and learning:
                for b in range(B):
                    # Positive birth/existence for real objects, negative for false-positive tracks.
                    mask_birth = corrected_observed[b] & correct_identity[b] & (alive[b, t] | false_track[b])
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
                            birth_policy["existence"], alive[b, t, mask_birth].float()
                        )
                        values["al_lifecycle"] = values["al_lifecycle"] + birth_loss / (B * learning_frames)
                        policy_examples += int(mask_birth.sum())
        if not learning:
            state, memory, anchor = state.detached(), memory.detach(), anchor.detach()
            camera_state = camera_state.detach()
            if use_al:
                agents = [agent.detached() for agent in agents]
            continue
        values["kl"] = values["kl"] + masked_mean(kl.clamp_min(c.free_nats), corrected_observed) / learning_frames
        anchors.append((t, state, memory, anchor, camera_state))
        if use_al:
            life_snapshots[t] = [agent.detached() for agent in agents]
        # Open-loop priors never consume future object observations or future camera evidence.
        imagined = state
        for k in range(1, min(c.horizon, T - 1 - t) + 1):
            weight = c.rollout_decay ** (k - 1)
            imagined_camera = model.camera.imagine(camera_state)
            imagined, forecast = model.world.predict(
                imagined, imagined_camera, batch["delta"][:, t + k], sample=model.training
            )
            supervised = state.valid & present[:, t + k] & real
            values["world"] = values["world"] + weight * masked_mean(
                motion_nll(forecast, boxes[:, t + k]), supervised
            )
            values["world"] = values["world"] + 0.2 * weight * masked_mean(
                1 - (forecast["feature"] * features[t + k].detach()).sum(-1), supervised
            )
            values["visibility"] = values["visibility"] + weight * masked_mean(
                F.binary_cross_entropy_with_logits(
                    forecast["visibility"], present[:, t + k].float(), reduction="none"
                ),
                state.valid & real,
            )
            rollout_weight += weight
    values["world"] = values["world"] / max(1e-8, rollout_weight) if rollout_weight else values["world"]
    values["visibility"] = (
        values["visibility"] / max(1e-8, rollout_weight) if rollout_weight else values["visibility"]
    )
    probes = 0
    if step % c.probe_every == 0 and anchors:
        t, saved_state, saved_memory, saved_anchor, _ = anchors[0]
        for b in range(B):
            observations = [
                (
                    slot_boxes[s][b],
                    slot_features[s][b],
                    observed_sequence[b, s],
                    observed_motions[s][b : b + 1],
                    batch["delta"][b : b + 1, s],
                    slot_confidence[s][b],
                    slot_classes[s][b],
                )
                for s in range(t + 1, min(T, t + 1 + c.probe_horizon))
            ]
            result = intervention_loss(
                model,
                saved_state.select(slice(b, b + 1)),
                saved_memory[b],
                track_classes[b],
                observations,
                collect=use_al,
                anchor=saved_anchor[b],
                agents=life_snapshots[t][b] if use_al else None,
                legacy=legacy,
            )
            effect, stability, count = result[:3]
            if use_al:
                learned_effect, actor, examples = effect_supervision(
                    model,
                    result[3],
                    life_snapshots[t][b],
                    saved_memory[b].detach(),
                    saved_anchor[b].detach(),
                    track_classes[b],
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
        + c.al_weight
        * (values["al_lifecycle"] + values["al_memory"] + values["al_effect"] + 0.1 * values["al_actor"])
    )
    if legacy:
        loss = loss + c.effect_weight * values["effect"] + c.stability_weight * values["stability"]
    metrics = {key: float(value.detach()) for key, value in values.items()}
    metrics.update(
        loss=float(loss.detach()),
        reid=float(reid.detach()),
        teacher_forcing=forcing,
        proposal_probability=proposal_probability,
        distractors=distractor_count,
        false_tracks=sum(len(x) for x in tracklets),
        probes=probes,
        al_policy_examples=policy_examples,
        al_critic_examples=critic_examples,
    )
    return loss, metrics

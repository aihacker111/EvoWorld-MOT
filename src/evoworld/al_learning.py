"""Supervision for life cycles, actual ROI acquisition, repair and intervention value."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.func import functional_call
from evoworld.geometry import pair_iou, sample_features
from evoworld.models.life import uncertainty_forecast
from evoworld.planning import ComputeBudget, allocate_planning, imagine, pooled_features


def mean_masked(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def censored_labels(batch):
    if "alive" in batch:
        return batch["alive"], batch["life_known"]
    # Handcrafted test batches have no sequence extent: only positives are known.
    present = batch["present"]
    future = present.flip(1).int().cumsum(1).flip(1) > 0
    past = present.int().cumsum(1) > 0
    alive = future & past
    return alive, alive


def observation_masks(present, config):
    """Independent dropout plus bursts of missing observations of random length.

    Deployed agents stay dormant for tens of frames, so bursts cover 1..al_burst_length
    frames and may repeat within one clip.
    """
    result = present & (torch.rand_like(present.float()) >= config.synthetic_dropout)
    if config.tracker_mode != "al":
        return result
    B, T, N = result.shape
    low = max(1, config.burn_in)
    if low >= T:
        return result
    for b in range(B):
        for i in range(N):
            for _ in range(config.al_burst_count):
                if torch.rand((), device=result.device) >= config.al_burst_probability:
                    continue
                start = int(torch.randint(low, T, (), device=result.device))
                length = int(torch.randint(1, config.al_burst_length + 1, (), device=result.device))
                result[b, start : min(T, start + length), i] = False
    return result


def corrupt_memory(memory, valid, probability):
    if probability <= 0:
        return memory
    mask = valid & (torch.rand_like(valid.float()) < probability)
    noisy = F.normalize(torch.randn_like(memory), dim=-1)
    return torch.where(mask[..., None], F.normalize(0.1 * memory + 0.9 * noisy, dim=-1), memory)


def candidate_targets(observed, candidates):
    """Track i's correct column is its own slot when observed, else the null column."""
    target = torch.arange(len(observed), device=observed.device)
    return torch.where(observed, target, candidates)


def frame_supervision(
    model,
    state,
    forecast,
    agents,
    scores,
    memory,
    anchor,
    candidates,
    dense,
    track_classes,
    observed,
    gt_alive,
    known,
    camera_prior,
    delta,
    false_track=None,
):
    """Lifecycle, existence and acquisition targets on the full candidate set.

    `candidates` = (boxes, features, confidence, classes, available) for the slot
    measurements followed by detector false-positive distractors.
    """
    c = model.config
    boxes, features, confidence, classes, available = candidates
    stats = agents.statistics(state, forecast, scores, c)
    policies = model.al.agents(state.h[0], stats)
    eligible = state.valid[0]
    label_mask = known & eligible
    existence = mean_masked(
        F.binary_cross_entropy_with_logits(policies["existence"], gt_alive.float(), reduction="none"),
        label_mask,
    )
    target_phase = torch.where(observed, 0, 1)
    ended = ~gt_alive & known & (agents.age >= c.al_exit_grace)
    if false_track is not None:
        # A track born from a detector false positive should end, even while it is "seen".
        ended = ended | (false_track & known)
    target_phase = torch.where(ended, 2, target_phase)
    phase = mean_masked(F.cross_entropy(policies["lifecycle"], target_phase, reduction="none"), label_mask)
    # Measure the value of an actual additional image operation on the same candidate set.
    # The benefit is a detached regression target: no graph is kept for this second scoring.
    with torch.no_grad():
        roi = pooled_features(dense, boxes[None])[0]
        roi_scores = model.association.scores(
            state, memory, anchor, boxes, roi, track_classes, classes, forecast, confidence
        )
        roi_scores = torch.where(torch.isfinite(roi_scores), roi_scores, roi_scores.new_full((), -1e4))
        roi_scores = roi_scores.masked_fill(~available[None], -1e4)
        target = candidate_targets(observed, len(boxes))
        null = scores.new_full((len(observed), 1), c.match_threshold)
        center_ce = F.cross_entropy(torch.cat((scores, null), -1), target, reduction="none")
        roi_ce = F.cross_entropy(torch.cat((roi_scores, null), -1), target, reduction="none")
        benefit = (center_ce - roi_ce).clamp(0, 4)
    acquisition = mean_masked(
        F.smooth_l1_loss(policies["resource"][:, 1], benefit, reduction="none"), eligible
    )
    budget = ComputeBudget(c.al_compute_budget)
    selected = allocate_planning(
        policies["resource"][:, 0].detach(), eligible & (agents.credit >= c.al_plan_horizon), c, budget
    )
    cache = imagine(model, state, camera_prior, selected, c.al_plan_horizon, delta)
    agents.credit = agents.credit.clone()
    agents.credit[selected] -= c.al_plan_horizon
    summary = cache.summary(memory)
    return existence + phase + 0.2 * acquisition, stats, summary, policies, int(label_mask.sum()), roi


def ghost_boxes(gt_boxes, present, count, avoid=None):
    """Background boxes whose sizes follow the clip's real object sizes.

    A fixed large ghost box is trivially separable by size, teaching a shortcut rather
    than existence.
    """
    device = gt_boxes.device
    sizes = gt_boxes[present][:, 2:] if present.any() else gt_boxes.new_tensor([[0.03, 0.04]])
    reference = gt_boxes[present] if present.any() else gt_boxes.new_zeros(0, 4)
    if avoid is not None and len(avoid):
        reference = torch.cat((reference, avoid))
    result = []
    for _ in range(8 * max(1, count)):
        if len(result) >= count:
            break
        size = sizes[torch.randint(len(sizes), (), device=device)] * (0.7 + 0.6 * torch.rand(2, device=device))
        size = size.clamp(0.005, 0.5)
        center = size / 2 + torch.rand(2, device=device) * (1 - size)
        box = torch.cat((center, size))
        if not len(reference) or float(pair_iou(box[None], reference).max()) < 0.05:
            result.append(box)
    return torch.stack(result) if result else gt_boxes.new_zeros(0, 4)


def ghost_supervision(model, dense, boxes, confidence):
    """Real background crops or detector false positives as negative tentative agents."""
    if not len(boxes):
        return dense.sum() * 0
    k = len(boxes)
    feature = sample_features(dense, boxes[None])
    initial = model.world.initial(dense, 1, k)
    born, _ = model.world.correct(
        initial, boxes[None], feature, torch.ones(1, k, dtype=torch.bool, device=dense.device), dense.new_ones(1)
    )
    # No GT is fed to these inputs; labels only declare that this crop is not a tracked object.
    stats = model.al.birth_statistics(confidence, model.config)
    policy = model.al.agents(born.h[0], stats)
    pair = model.association.context(
        born.h[0],
        feature[0],
        feature[0],
        feature[0],
        boxes,
        boxes,
        confidence,
        confidence.new_zeros(k),
        confidence.new_zeros(k),
    )
    quality, _ = model.al.proposal(
        model.al.context(pair, stats, stats.new_zeros(k, 4)), model.association.gate(pair)
    )
    return (
        F.binary_cross_entropy_with_logits(policy["existence"], torch.zeros_like(policy["existence"]))
        + F.cross_entropy(policy["lifecycle"], torch.full((k,), 2, dtype=torch.long, device=dense.device))
        + 0.1 * policy["resource"].square().mean()
        + 0.2 * F.binary_cross_entropy_with_logits(quality, torch.zeros_like(quality))
    )


def memory_supervision(
    model,
    pair_context,
    stats,
    summary,
    prior_gate,
    memory,
    measurement,
    observed,
    correct,
    negatives=None,
):
    """Repair on correct observations, conservative writes otherwise.

    `negatives` = (pair_context, stats, summary, prior_gate) for track/false-positive
    pairs: their match quality and write gate must both be low.
    """
    context = model.al.context(pair_context, stats, summary)
    quality, gate = model.al.proposal(context, prior_gate)
    discrepancy = (1 - (memory * measurement).sum(-1)).clamp(0, 1)
    target_gate = correct.float() * (0.1 + 0.7 * discrepancy.detach())
    quality_loss = mean_masked(
        F.binary_cross_entropy_with_logits(quality, correct.float(), reduction="none"), observed
    )
    gate_logits = torch.logit(gate.float().clamp(1e-5, 1 - 1e-5))
    gate_loss = mean_masked(
        F.binary_cross_entropy_with_logits(gate_logits, target_gate.float(), reduction="none"), observed
    )
    loss = quality_loss + 0.2 * gate_loss
    if negatives is not None and len(negatives[0]):
        negative_quality, negative_gate = model.al.proposal(
            model.al.context(negatives[0], negatives[1], negatives[2]), negatives[3]
        )
        logits = torch.logit(negative_gate.float().clamp(1e-5, 1 - 1e-5))
        loss = loss + 0.5 * (
            F.binary_cross_entropy_with_logits(negative_quality, torch.zeros_like(negative_quality))
            + 0.2 * F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        )
    return gate, loss


def sample_contexts(model, samples, memory, anchor, agents, stats, observations):
    """Rebuild critic pair contexts exactly as the deployed tracker builds them."""
    association = model.association
    state = samples["state"]
    boxes, features = observations[0][:2]
    confidence = samples["confidence"]
    i, j, types = samples["tracks"], samples["detections"], samples["types"]
    wait = types == 1
    measurement = torch.where(wait[:, None], memory[i], features[j])
    measured_box = torch.where(wait[:, None], state.box[0, i], boxes[j])
    measured_confidence = torch.where(wait, torch.zeros_like(confidence[j]), confidence[j])
    return association.context(
        state.h[0, i],
        memory[i],
        anchor[i],
        measurement,
        state.box[0, i],
        measured_box,
        measured_confidence,
        agents.age[i],
        stats[i, 9] * 5,
    )


def effect_supervision(model, samples, agents, memory, anchor, classes, camera_prior, observations):
    if samples is None:
        return memory.sum() * 0, memory.sum() * 0, 0
    c = model.config
    state, forecast = samples["state"], samples["forecast"]
    from evoworld.interventions import unpack

    boxes, features, present, _, delta, confidence, detection_classes = unpack(observations[0], classes)
    agents = agents.detached().predict(forecast, delta, c, state.box[0])
    forecast = uncertainty_forecast(forecast, agents)
    scores = model.association.scores(
        state, memory, anchor, boxes, features, classes, detection_classes, forecast, confidence
    )
    scores = torch.where(torch.isfinite(scores), scores, scores.new_full((), -1e4))
    scores = scores.masked_fill(~present[None], -1e4)
    stats = agents.statistics(state, forecast, scores, c).detach()
    policy = model.al.agents(state.h[0].detach(), stats)
    budget = ComputeBudget(c.al_compute_budget)
    index = allocate_planning(policy["resource"][:, 0].detach(), state.valid[0], c, budget)
    cache = imagine(model, state, camera_prior, index, c.al_plan_horizon)
    summary = cache.summary(memory)
    i, j, types = samples["tracks"], samples["detections"], samples["types"]
    pair = sample_contexts(model, samples, memory, anchor, agents, stats, observations).detach()
    action_gate = samples["gates"]
    base = model.al.context(pair, stats[i], summary[i])
    measurement = torch.where((types == 0)[:, None], features[j], memory[i])
    affinity = cache.affinities(memory[i], measurement, action_gate, i)
    context = model.al.action_context(base, action_gate, affinity)
    context[:, -3:] = F.one_hot(types, 3).to(context)
    predicted = model.al.effects(context)
    target = samples["targets"].detach()
    variance = predicted[:, 2].square()
    regression = (0.5 * (predicted[:, 0] - target[:, 0]).square() / variance + 0.5 * variance.log()).mean()
    gain = F.smooth_l1_loss(predicted[:, 1], target[:, 1])
    difference = target[:, None, 0] - target[None, :, 0]
    same = (i[:, None] == i[None]) & (difference.abs() > 0.1)
    ranking = F.softplus(-(predicted[:, None, 0] - predicted[None, :, 0]) * difference.sign())
    ranking = ranking[same].mean() if same.any() else predicted.sum() * 0
    match = types == 0
    quality, proposed_gate = model.al.proposal(base, model.association.gate(pair))
    correct = (i == j).float()
    quality_loss = mean_masked(F.binary_cross_entropy_with_logits(quality, correct, reduction="none"), match)
    actor = predicted.sum() * 0
    if int(model.al.critic_examples) >= c.al_critic_min_examples:
        # Only the proposal policy learns on this path, preventing a critic from reducing its own penalty.
        actor_base = base.detach()
        _, gate = model.al.proposal(actor_base, model.association.gate(pair.detach()))
        actor_affinity = cache.affinities(memory[i].detach(), measurement.detach(), gate, i)
        actor_context = model.al.action_context(actor_base, gate, actor_affinity)
        raw = functional_call(
            model.al.critic, {k: v.detach() for k, v in model.al.critic.named_parameters()}, (actor_context,)
        )
        risk = raw[:, 0].sigmoid() * model.al.horizon + F.relu(F.softplus(raw[:, 1]) - c.stability_limit)
        actor = mean_masked(risk, match)
    # Target potential value of computation from the same paired replay, not an arbitrary age threshold.
    resource_losses = []
    for identity in i.unique().tolist():
        group = i == identity
        baseline_det = int(samples["baseline"][group][0])
        baseline = group & (
            ((types == 1) & (baseline_det < 0))
            | ((types == 0) & (j == baseline_det) & ((action_gate - 0.25).abs() < 0.01))
        )
        if baseline.any():
            value = (target[baseline, 0].mean() - target[group, 0].min()).clamp_min(0)
            resource_losses.append(F.smooth_l1_loss(policy["resource"][identity, 0], value))
    resource_loss = torch.stack(resource_losses).mean() if resource_losses else predicted.sum() * 0
    return regression + gain + 0.2 * ranking + 0.2 * quality_loss + 0.2 * resource_loss, actor, len(i)

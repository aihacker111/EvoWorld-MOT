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
    result = present & (torch.rand_like(present.float()) >= config.synthetic_dropout)
    if config.tracker_mode != "al":
        return result
    B, T, N = result.shape
    for b in range(B):
        for i in range(N):
            if torch.rand((), device=result.device) < config.al_burst_probability:
                start = int(
                    torch.randint(
                        max(1, config.burn_in), max(config.burn_in + 1, T), (), device=result.device
                    )
                )
                result[b, start : min(T, start + config.al_burst_length), i] = False
    return result


def corrupt_memory(memory, valid, probability):
    if probability <= 0:
        return memory
    mask = valid & (torch.rand_like(valid.float()) < probability)
    noisy = F.normalize(torch.randn_like(memory), dim=-1)
    return torch.where(mask[..., None], F.normalize(0.1 * memory + 0.9 * noisy, dim=-1), memory)


def frame_supervision(
    model,
    state,
    forecast,
    agents,
    scores,
    memory,
    anchor,
    boxes,
    features,
    dense,
    classes,
    observed,
    gt_alive,
    known,
    camera_prior,
    delta,
):
    c = model.config
    stats = agents.statistics(state, forecast, scores, c)
    policies = model.al.agents(state.h[0], stats)
    eligible = state.valid[0]
    label_mask = known & eligible
    existence = mean_masked(
        F.binary_cross_entropy_with_logits(policies["existence"], gt_alive.float(), reduction="none"),
        label_mask,
    )
    target_phase = torch.where(observed, 0, 1)
    target_phase = torch.where(~gt_alive & known & (agents.age >= c.al_exit_grace), 2, target_phase)
    phase = mean_masked(F.cross_entropy(policies["lifecycle"], target_phase, reduction="none"), label_mask)
    # Measure the value of an actual additional image operation, rather than naming a counter "energy".
    roi = pooled_features(dense, boxes[None])[0]
    roi_scores = model.association.scores(state, memory, anchor, boxes, roi, classes, classes, forecast)
    available = observed[None]
    center_scores = scores.masked_fill(~available, -1e4)
    roi_scores = roi_scores.masked_fill(~available, -1e4)
    target = torch.arange(len(boxes), device=boxes.device)
    target = torch.where(observed, target, len(boxes))
    null = scores.new_full((len(boxes), 1), c.match_threshold)
    center_ce = F.cross_entropy(torch.cat((center_scores, null), -1), target, reduction="none")
    roi_ce = F.cross_entropy(torch.cat((roi_scores, null), -1), target, reduction="none")
    benefit = (center_ce - roi_ce).detach().clamp(0, 4)
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


def ghost_supervision(model, dense, gt_boxes, present):
    """A real background crop supplies a negative, tentative agent during training."""
    candidate = None
    for _ in range(8):
        box = torch.cat((torch.rand(2, device=dense.device) * 0.8 + 0.1, dense.new_tensor([0.08, 0.12])))
        if not present.any() or float(pair_iou(box[None], gt_boxes[present]).max()) < 0.05:
            candidate = box
            break
    if candidate is None:
        return dense.sum() * 0
    feature = sample_features(dense, candidate[None, None])
    initial = model.world.initial(dense, 1, 1)
    born, _ = model.world.correct(
        initial,
        candidate[None, None],
        feature,
        torch.ones(1, 1, dtype=torch.bool, device=dense.device),
        dense.new_ones(1),
    )
    # No GT is fed to these inputs; labels only declare that this crop is not a tracked object.
    confidence = 0.5 + 0.45 * torch.rand(1, device=dense.device)
    stats = model.al.birth_statistics(confidence, model.config)
    policy = model.al.agents(born.h[0], stats)
    pair = model.association.context(
        born.h[0],
        feature[0],
        feature[0],
        feature[0],
        candidate[None],
        candidate[None],
        confidence,
        confidence.new_zeros(1),
        confidence.new_zeros(1),
    )
    quality, _ = model.al.proposal(
        model.al.context(pair, stats, stats.new_zeros(1, 4)), model.association.gate(pair)
    )
    return (
        F.binary_cross_entropy_with_logits(policy["existence"], torch.zeros_like(policy["existence"]))
        + F.cross_entropy(policy["lifecycle"], torch.full((1,), 2, dtype=torch.long, device=dense.device))
        + 0.1 * policy["resource"].square().mean()
        + 0.2 * F.binary_cross_entropy_with_logits(quality, torch.zeros_like(quality))
    )


def memory_supervision(
    model, pair_context, stats, summary, prior_gate, memory, measurement, observed, correct
):
    context = model.al.context(pair_context, stats, summary)
    quality, gate = model.al.proposal(context, prior_gate)
    # Correct observations should repair corrupted memory, with conservative writes otherwise.
    discrepancy = (1 - (memory * measurement).sum(-1)).clamp(0, 1)
    target_gate = correct.float() * (0.1 + 0.7 * discrepancy.detach())
    quality_loss = mean_masked(
        F.binary_cross_entropy_with_logits(quality, correct.float(), reduction="none"), observed
    )
    gate_logits = torch.logit(gate.float().clamp(1e-5, 1 - 1e-5))
    gate_loss = mean_masked(
        F.binary_cross_entropy_with_logits(gate_logits, target_gate.float(), reduction="none"), observed
    )
    return gate, quality_loss + 0.2 * gate_loss


def effect_supervision(model, samples, agents, memory, classes, camera_prior, observations):
    if samples is None:
        return memory.sum() * 0, memory.sum() * 0, 0
    c = model.config
    state, forecast = samples["state"], samples["forecast"]
    boxes, features, present = observations[0][:3]
    agents = agents.detached().predict(forecast, observations[0][4], c)
    forecast = uncertainty_forecast(forecast, agents)
    scores = model.association.scores(state, memory, memory, boxes, features, classes, classes, forecast)
    scores = scores.masked_fill(~present[None], -1e4)
    stats = agents.statistics(state, forecast, scores, c).detach()
    policy = model.al.agents(state.h[0].detach(), stats)
    budget = ComputeBudget(c.al_compute_budget)
    index = allocate_planning(policy["resource"][:, 0].detach(), state.valid[0], c, budget)
    cache = imagine(model, state, camera_prior, index, c.al_plan_horizon)
    summary = cache.summary(memory)
    i, j, types = samples["tracks"], samples["detections"], samples["types"]
    pair = samples["contexts"].detach().clone()
    action_gate = pair[:, model.al._gate_index].clone()
    pair[:, model.al._gate_index] = 0.0
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

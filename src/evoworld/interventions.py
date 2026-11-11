"""Paired tracker-state interventions on one cached, causally shared observation stream."""

import torch
import torch.nn.functional as F
from torch.func import functional_call
from evoworld.geometry import warp_boxes
from evoworld.classes import same_group
from evoworld.models.association import assign, assign_with_null, blend_memory
from evoworld.models.life import uncertainty_forecast
from evoworld.models.world import damp_missing


def unpack(observation, classes):
    boxes, features, present, affine, delta = observation[:5]
    confidence = observation[5] if len(observation) > 5 else features.new_ones(len(boxes))
    detection_classes = observation[6] if len(observation) > 6 else classes
    return boxes, features, present, affine, delta, confidence, detection_classes


def policy_assignment(model, predicted, forecast, agents, memory, anchor, scores, boxes, features, confidence):
    """Deployed AL arbitration without planning/critic: learned quality, null and gate."""
    c = model.config
    n, m = scores.shape
    finite = torch.where(torch.isfinite(scores), scores, scores.new_full((), -1e4))
    stats = agents.statistics(predicted, forecast, finite, c)
    policies = model.al.agents(predicted.h[0], stats)
    phase = policies["lifecycle"].log_softmax(-1)
    null = scores.new_full((n,), c.match_threshold) + 0.3 * (phase[:, 1] - phase[:, 0]).clamp(-3, 3)
    if not m:
        return [], scores.new_zeros(n, 0), policies
    margin = stats[:, 9] * 5
    pair = model.association.context(
        predicted.h[0, :, None].expand(-1, m, -1),
        memory[:, None].expand(-1, m, -1),
        anchor[:, None].expand(-1, m, -1),
        features[None].expand(n, -1, -1),
        predicted.box[0, :, None].expand(-1, m, -1),
        boxes[None].expand(n, -1, -1),
        confidence[None].expand(n, -1),
        agents.age[:, None].expand(-1, m),
        margin[:, None].expand(-1, m),
    )
    base = model.al.context(pair, stats[:, None].expand(-1, m, -1), stats.new_zeros(n, m, 4))
    quality, gate = model.al.proposal(base, model.association.gate(pair))
    utility = scores + 0.3 * F.logsigmoid(quality)
    utility = utility.masked_fill(quality.sigmoid() < c.al_match_probability, -torch.inf)
    return assign_with_null(utility, null), gate, policies


@torch.no_grad()
def replay(model, state, memory, anchor, classes, observations, forced, gate, target, agents=None):
    """Replay one intervention.  With `agents`, continuation follows the AL policy (on-policy)."""
    c = model.config
    state = state.clone()
    memory, anchor = memory.clone(), anchor.clone()
    errors = memory.new_zeros(())
    for t, observation in enumerate(observations):
        boxes, features, present, affine, delta, confidence, detection_classes = unpack(observation, classes)
        previous = warp_boxes(state.box, affine)
        predicted, forecast = model.world.predict(state, affine, delta)
        if agents is not None:
            agents = agents.predict(forecast, delta, c, predicted.box[0])
            forecast = uncertainty_forecast(forecast, agents)
        indices = present.nonzero().flatten()
        scores = model.association.scores(
            predicted,
            memory,
            anchor,
            boxes[indices],
            features[indices],
            classes,
            detection_classes[indices],
            forecast,
            confidence[indices],
        )
        gates, policies = None, None
        if t == 0:
            lookup = {int(j): k for k, j in enumerate(indices)}
            pairs = [(i, lookup[j]) for i, j in forced if j in lookup]
        elif agents is not None:
            pairs, gates, policies = policy_assignment(
                model,
                predicted,
                forecast,
                agents,
                memory,
                anchor,
                scores,
                boxes[indices],
                features[indices],
                confidence[indices],
            )
        else:
            pairs = assign(scores, c.match_threshold)
        measured = predicted.box.clone()
        observed = torch.zeros_like(predicted.valid)
        appearances = memory.clone()
        measured_confidence = memory.new_zeros(len(memory))
        target_assignment = None
        for i, k in pairs:
            j = int(indices[k])
            measured[0, i], appearances[i], observed[0, i] = boxes[j], features[j], True
            measured_confidence[i] = confidence[j]
            if t == 0 and i == target:
                update = gate
            elif gates is not None:
                update = gates[i, k]
            else:
                update = memory.new_tensor(0.15)
            memory[i] = blend_memory(memory[i], features[j], update)
            if i == target:
                target_assignment = j
        if present[target] and target_assignment != target:
            errors += 1
        state, _ = model.world.correct(predicted, measured, appearances[None], observed, delta, previous)
        state = damp_missing(state, observed, delta, c.dormant_velocity_decay)
        if agents is not None:
            alive = (
                policies["existence"].sigmoid() if policies is not None else agents.existence
            )
            agents = agents.observe(
                observed[0], measured_confidence, measured[0], measured_confidence * 0, alive, True, c
            )
    return errors, state, memory


def intervention_loss(
    model,
    state,
    memory,
    classes,
    observations,
    max_targets=2,
    collect=False,
    anchor=None,
    agents=None,
    legacy=True,
):
    association = model.association
    zero = memory.sum() * 0
    if not observations:
        return (zero, zero, 0, None) if collect else (zero, zero, 0)
    state, memory = state.detached(), memory.detach()
    anchor = memory if anchor is None else anchor.detach()
    detached = [tuple(x.detach() for x in row) for row in observations]
    boxes, features, present, affine, delta, confidence, detection_classes = unpack(detached[0], classes)
    predicted, forecast = model.world.predict(state, affine, delta)
    predicted = predicted.detached()
    forecast = {k: v.detach() for k, v in forecast.items()}
    indices = (present & state.valid[0]).nonzero().flatten().tolist()
    scores = association.scores(
        predicted, memory, anchor, boxes, features, classes, detection_classes, forecast, confidence
    ).detach()
    eligible_scores = scores.masked_fill(~(present & state.valid[0])[None], -1e4)
    baseline_assignment = dict(assign(eligible_scores, model.config.match_threshold))
    compatible = same_group(classes, classes, model.config)
    on_policy = agents if (model.config.on_policy_replay and agents is not None) else None
    contexts, targets, tracks, detections, types, baselines, gates, gate_objectives = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    order = torch.randperm(len(indices), device=memory.device).tolist()
    for selection in order[:max_targets]:
        i = indices[selection]
        others = [j for j in indices if j != i and compatible[i, j]]
        others.sort(key=lambda j: float(scores[i, j]), reverse=True)
        baseline_pairs = [(j, j) for j in indices]
        baseline = replay(
            model, state, memory, anchor, classes, detached, baseline_pairs, memory.new_tensor(0.15), i, on_policy
        )
        for j in [i] + others[:1]:
            forced = [(a, j if a == i else i if a == j else a) for a in indices]
            context = association.context(
                predicted.h[0, i],
                memory[i],
                anchor[i],
                features[j],
                predicted.box[0, i],
                boxes[j],
                confidence[j],
                memory.new_tensor(0),
                memory.new_tensor(0.0),
            )
            initial_distance = (features[j] - features[i]).norm().clamp_min(0.1)
            for value in (0.0, 0.25, 1.0):
                action = memory.new_tensor(value)
                result = replay(model, state, memory, anchor, classes, detached, forced, action, i, on_policy)
                gain = ((result[2][i] - baseline[2][i]).norm() / initial_distance).clamp_max(10)
                contexts.append(torch.cat((context[:-1], action[None])))
                targets.append(torch.stack((result[0], gain)))
                tracks.append(i)
                detections.append(j)
                types.append(0)
                gates.append(value)
                baselines.append(baseline_assignment.get(i, -1))
            if legacy:
                gate = association.gate(context)
                gated_context = torch.cat((context[:-1], gate[None]))
                raw = functional_call(
                    association.effect,
                    {k: v.detach() for k, v in association.effect.named_parameters()},
                    (gated_context,),
                )
                if association.effect_examples >= model.config.effect_min_examples:
                    gate_objectives.append(
                        raw[0].sigmoid() * association.horizon
                        + F.relu(F.softplus(raw[1]) - model.config.stability_limit)
                    )
        if collect:
            # Explicit wait alternative teaches the critic that survival alone is not the objective.
            forced = [(a, a) for a in indices if a != i]
            result = replay(
                model, state, memory, anchor, classes, detached, forced, memory.new_tensor(0.0), i, on_policy
            )
            context = association.context(
                predicted.h[0, i],
                memory[i],
                anchor[i],
                memory[i],
                predicted.box[0, i],
                predicted.box[0, i],
                memory.new_tensor(0.0),
                memory.new_tensor(0),
                memory.new_tensor(0.0),
            )
            contexts.append(context)
            gain = ((result[2][i] - baseline[2][i]).norm() / 0.1).clamp_max(10)
            targets.append(torch.stack((result[0], gain)))
            tracks.append(i)
            detections.append(i)
            types.append(1)
            gates.append(0.0)
            baselines.append(baseline_assignment.get(i, -1))
    if not contexts:
        return (zero, zero, 0, None) if collect else (zero, zero, 0)
    contexts, targets = torch.stack(contexts), torch.stack(targets)
    if legacy:
        prediction = association.effects(contexts)
        variance = prediction[:, 2].square()
        cost = (0.5 * (prediction[:, 0] - targets[:, 0]).square() / variance + 0.5 * variance.log()).mean()
        gain = F.smooth_l1_loss(prediction[:, 1], targets[:, 1])
        difference = targets[:, None, 0] - targets[None, :, 0]
        valid = difference.abs() > 0.1
        ranking = F.softplus(-(prediction[:, None, 0] - prediction[None, :, 0]) * difference.sign())
        ranking = ranking[valid].mean() if valid.any() else prediction.sum() * 0
        stability = torch.stack(gate_objectives).mean() if gate_objectives else prediction.sum() * 0
        result = (cost + gain + 0.2 * ranking, stability, len(contexts))
    else:
        result = (zero, zero, len(contexts))
    if not collect:
        return result

    def tensor(x, dtype=torch.long):
        return torch.tensor(x, device=memory.device, dtype=dtype)

    samples = dict(
        contexts=contexts.detach(),
        targets=targets.detach(),
        tracks=tensor(tracks),
        detections=tensor(detections),
        types=tensor(types),
        gates=tensor(gates, memory.dtype),
        baseline=tensor(baselines),
        confidence=confidence.detach(),
        state=predicted,
        forecast=forecast,
    )
    return (*result, samples)

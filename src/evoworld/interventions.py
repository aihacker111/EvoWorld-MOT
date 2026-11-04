"""Paired tracker-state interventions on one cached, causally shared observation stream."""

import torch
import torch.nn.functional as F
from torch.func import functional_call
from evoworld.geometry import warp_boxes
from evoworld.models.association import assign, blend_memory


@torch.no_grad()
def replay(model, state, memory, anchor, classes, observations, forced, gate, target):
    state = state.clone()
    memory, anchor = memory.clone(), anchor.clone()
    errors = memory.new_zeros(())
    for t, observation in enumerate(observations):
        boxes, features, present, affine, delta = observation[:5]
        previous = warp_boxes(state.box, affine)
        predicted, forecast = model.world.predict(state, affine, delta)
        indices = present.nonzero().flatten()
        scores = model.association.scores(
            predicted, memory, anchor, boxes[indices], features[indices], classes, classes[indices], forecast
        )
        if t == 0:
            lookup = {int(j): k for k, j in enumerate(indices)}
            pairs = [(i, lookup[j]) for i, j in forced if j in lookup]
        else:
            pairs = assign(scores, model.config.match_threshold)
        measured = predicted.box.clone()
        observed = torch.zeros_like(predicted.valid)
        appearances = memory.clone()
        target_assignment = None
        for i, k in pairs:
            j = int(indices[k])
            measured[0, i], appearances[i], observed[0, i] = boxes[j], features[j], True
            update = gate if t == 0 and i == target else memory.new_tensor(0.15)
            memory[i] = blend_memory(memory[i], features[j], update)
            if i == target:
                target_assignment = j
        if present[target] and target_assignment != target:
            errors += 1
        state, _ = model.world.correct(predicted, measured, appearances[None], observed, delta, previous)
    return errors, state, memory


def intervention_loss(model, state, memory, classes, observations, max_targets=2, collect=False):
    association = model.association
    zero = memory.sum() * 0
    if not observations:
        return (zero, zero, 0, None) if collect else (zero, zero, 0)
    state, memory = state.detached(), memory.detach()
    detached = [tuple(x.detach() for x in row) for row in observations]
    boxes, features, present, affine, delta = detached[0][:5]
    confidence = detached[0][5] if len(detached[0]) > 5 else memory.new_ones(len(boxes))
    predicted, forecast = model.world.predict(state, affine, delta)
    predicted = predicted.detached()
    forecast = {k: v.detach() for k, v in forecast.items()}
    indices = (present & state.valid[0]).nonzero().flatten().tolist()
    scores = association.scores(
        predicted, memory, memory, boxes, features, classes, classes, forecast
    ).detach()
    eligible_scores = scores.masked_fill(~(present & state.valid[0])[None], -1e4)
    baseline_assignment = dict(assign(eligible_scores, model.config.match_threshold))
    contexts, targets, tracks, detections, types, baselines, gate_objectives = [], [], [], [], [], [], []
    order = torch.randperm(len(indices), device=memory.device).tolist()
    for selection in order[:max_targets]:
        i = indices[selection]
        others = [j for j in indices if j != i and classes[j] == classes[i]]
        others.sort(key=lambda j: float(scores[i, j]), reverse=True)
        baseline_pairs = [(j, j) for j in indices]
        baseline = replay(
            model, state, memory, memory, classes, detached, baseline_pairs, memory.new_tensor(0.15), i
        )
        for j in [i] + others[:1]:
            forced = [(a, j if a == i else i if a == j else a) for a in indices]
            context = association.context(
                predicted.h[0, i],
                memory[i],
                memory[i],
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
                result = replay(model, state, memory, memory, classes, detached, forced, action, i)
                gain = ((result[2][i] - baseline[2][i]).norm() / initial_distance).clamp_max(10)
                contexts.append(torch.cat((context[:-1], action[None])))
                targets.append(torch.stack((result[0], gain)))
                tracks.append(i)
                detections.append(j)
                types.append(0)
                baselines.append(baseline_assignment.get(i, -1))
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
                model, state, memory, memory, classes, detached, forced, memory.new_tensor(0.0), i
            )
            context = association.context(
                predicted.h[0, i],
                memory[i],
                memory[i],
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
            baselines.append(baseline_assignment.get(i, -1))
    if not contexts:
        return (zero, zero, 0, None) if collect else (zero, zero, 0)
    contexts, targets = torch.stack(contexts), torch.stack(targets)
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
    if not collect:
        return result

    def tensor(x):
        return torch.tensor(x, device=memory.device, dtype=torch.long)

    samples = dict(
        contexts=contexts.detach(),
        targets=targets.detach(),
        tracks=tensor(tracks),
        detections=tensor(detections),
        types=tensor(types),
        baseline=tensor(baselines),
        state=predicted,
        forecast=forecast,
    )
    return (*result, samples)

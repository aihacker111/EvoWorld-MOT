"""Bounded ecology of persistent tracking hypotheses; no image or GT duplication."""

from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch import Tensor
from evoworld.geometry import sample_features, warp_boxes
from evoworld.models.world import ObjectState
from evoworld.models.association import assign, blend_memory


@dataclass
class Hypothesis:
    state: ObjectState
    memory: Tensor
    anchor: Tensor
    classes: Tensor
    ids: Tensor
    age: Tensor
    score: float = 0.0
    lineage: tuple = ()
    output: list | None = None

    def select(self, index):
        state = ObjectState(**{f.name: getattr(self.state, f.name)[:, index] for f in fields(self.state)})
        return Hypothesis(
            state,
            self.memory[index],
            self.anchor[index],
            self.classes[index],
            self.ids[index],
            self.age[index],
            self.score,
            self.lineage,
            self.output,
        )


class EcologyTracker:
    def __init__(self, model):
        self.model, self.config = model, model.config
        self.reset()

    def reset(self):
        self.population = []
        self.camera_state = None
        self.previous_camera = None
        self.previous_frame = None
        self.next_id = 1
        self.birth_registry = {}
        self.frame = 0

    def _empty(self, reference):
        state = self.model.world.initial(reference, 1, 0)
        e = reference.new_empty(0, self.config.embedding_dim)
        ids = torch.empty(0, dtype=torch.long, device=reference.device)
        return Hypothesis(state, e, e.clone(), ids, ids.clone(), ids.clone())

    def _birth_id(self, detection):
        key = (self.frame, detection)
        if key not in self.birth_registry:
            self.birth_registry[key] = self.next_id
            self.next_id += 1
        return self.birth_registry[key]

    @torch.no_grad()
    def step(
        self, detections: dict, dense: Tensor, camera: Tensor, frame_id: int, affine: Tensor | None = None
    ):
        """Accept only current observations. Boxes are normalized cxcywh, class zero-based."""
        if self.previous_frame is not None and frame_id <= self.previous_frame:
            raise ValueError("Frames must be strictly increasing; reset between sequences")
        self.frame = frame_id
        dt = camera.new_tensor([1 if self.previous_frame is None else frame_id - self.previous_frame])
        previous = camera if self.previous_camera is None else self.previous_camera
        self.camera_state, learned = self.model.camera.observe(camera, previous, self.camera_state)
        motion = learned if affine is None else affine.reshape(1, 2, 3).to(camera)
        boxes, scores, classes = (detections[k] for k in ("boxes", "scores", "classes"))
        features = sample_features(dense, boxes[None])[0]
        if not self.population:
            self.population = [self._empty(dense)]
        candidates = []
        for parent in self.population:
            if parent.ids.numel():
                predicted, forecast = self.model.world.predict(parent.state, motion, dt)
                pair_scores = self.model.association.scores(
                    predicted,
                    parent.memory,
                    parent.anchor,
                    boxes,
                    features,
                    parent.classes,
                    classes,
                    forecast,
                )
                baseline = assign(pair_scores, self.config.match_threshold)
                assignments = [baseline]
                uncertain = []
                for i, j in baseline:
                    row = pair_scores[i]
                    other = torch.cat((row[:j], row[j + 1 :], row.new_tensor([self.config.match_threshold])))
                    margin = float(row[j] - other.max())
                    if self.config.ecology_budget > 1 and margin < self.config.uncertain_margin:
                        uncertain.append((margin, i, j))
                for _, i, j in sorted(uncertain)[: self.config.mutation_edges]:
                    alternative = assign(pair_scores, self.config.match_threshold, (i, j))
                    if alternative not in assignments:
                        assignments.append(alternative)
                for ai, pairs in enumerate(assignments):
                    candidates.append(
                        self._correct(
                            parent,
                            predicted,
                            pairs,
                            pair_scores,
                            boxes,
                            features,
                            scores,
                            classes,
                            motion,
                            dt,
                            False,
                            ai,
                        )
                    )
                    if uncertain:
                        candidates.append(
                            self._correct(
                                parent,
                                predicted,
                                pairs,
                                pair_scores,
                                boxes,
                                features,
                                scores,
                                classes,
                                motion,
                                dt,
                                True,
                                ai,
                            )
                        )
            else:
                candidates.append(
                    self._correct(
                        parent,
                        parent.state,
                        [],
                        dense.new_empty(0, len(boxes)),
                        boxes,
                        features,
                        scores,
                        classes,
                        motion,
                        dt,
                        False,
                        0,
                    )
                )
        candidates.sort(key=lambda h: h.score, reverse=True)
        # Avoid clones crowding out a genuinely different assignment or memory policy.
        unique, keys = [], set()
        for candidate in candidates:
            signature = (tuple((row[0], row[1]) for row in candidate.output), candidate.lineage[-2:])
            if signature not in keys:
                keys.add(signature)
                unique.append(candidate)
            if len(unique) == self.config.ecology_budget:
                break
        self.population = unique
        self.previous_frame, self.previous_camera = frame_id, camera
        # Birth registry is only needed while constructing this frame's descendants.
        self.birth_registry.clear()
        return self.population[0].output

    def _correct(
        self,
        parent,
        predicted,
        pairs,
        pair_scores,
        boxes,
        features,
        scores,
        classes,
        motion,
        dt,
        freeze,
        alternative,
    ):
        n, m = len(parent.ids), len(boxes)
        observed = torch.zeros(1, n, dtype=torch.bool, device=boxes.device)
        measurements = predicted.box.clone()
        appearance = parent.memory.clone()
        memory = parent.memory.clone()
        age = parent.age + dt.long()[0]
        score = parent.score * 0.98
        output = []
        matched = set()
        for i, j in pairs:
            matched.add(j)
            observed[0, i] = True
            measurements[0, i] = boxes[j]
            appearance[i] = features[j]
            age[i] = 0
            row = pair_scores[i]
            others = torch.cat((row[:j], row[j + 1 :], row.new_tensor([self.config.match_threshold])))
            margin = row[j] - others.max()
            context = self.model.association.context(
                predicted.h[0, i],
                parent.memory[i],
                parent.anchor[i],
                features[j],
                predicted.box[0, i],
                boxes[j],
                scores[j],
                parent.age[i],
                margin,
            )
            gate = self.model.association.gate(context)
            gate = torch.zeros_like(gate) if freeze else gate
            context = torch.cat((context[:-1], gate[None]), -1)
            if self.model.association.effect_examples >= self.config.effect_min_examples:
                effect = self.model.association.effects(context)
                risk = effect[0] + self.config.effect_margin * effect[2]
                score -= self.config.effect_score_weight * float(risk) / max(1, n)
            memory[i] = blend_memory(parent.memory[i], features[j], gate)
            score += float(pair_scores[i, j] - self.config.match_threshold) / max(1, n)
            if scores[j] >= self.config.conf_threshold:
                output.append((int(parent.ids[i]), j, boxes[j], float(scores[j]), int(classes[j])))
        score -= 0.1 * (n - len(pairs)) / max(1, n)
        if n:
            state, _ = self.model.world.correct(
                predicted, measurements, appearance[None], observed, dt, warp_boxes(parent.state.box, motion)
            )
        else:
            state = predicted
        child = Hypothesis(
            state,
            memory,
            parent.anchor.clone(),
            parent.classes.clone(),
            parent.ids.clone(),
            age,
            score,
            (parent.lineage + ((alternative, int(freeze)),))[-4:],
            output,
        )
        keep = child.age <= self.config.max_age
        child = child.select(keep)
        births = [j for j in range(m) if j not in matched and scores[j] >= self.config.birth_threshold]
        if births:
            index = torch.tensor(births, device=boxes.device)
            initial = self.model.world.initial(boxes, 1, len(births))
            born, _ = self.model.world.correct(
                initial,
                boxes[index][None],
                features[index][None],
                torch.ones(1, len(births), dtype=torch.bool, device=boxes.device),
                dt,
            )
            child.state = ObjectState(
                **{
                    f.name: torch.cat((getattr(child.state, f.name), getattr(born, f.name)), 1)
                    for f in fields(born)
                }
            )
            child.memory = torch.cat((child.memory, features[index]), 0)
            child.anchor = torch.cat((child.anchor, features[index]), 0)
            child.classes = torch.cat((child.classes, classes[index]), 0)
            ids = child.ids.new_tensor([self._birth_id(j) for j in births])
            child.ids = torch.cat((child.ids, ids), 0)
            child.age = torch.cat((child.age, torch.zeros_like(ids)), 0)
            child.output = output + [
                (int(identity), j, boxes[j], float(scores[j]), int(classes[j]))
                for identity, j in zip(ids, births)
            ]
        return child

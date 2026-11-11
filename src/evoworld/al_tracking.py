"""Per-identity life agents with learned maintenance and bounded acquisition.

Association runs in two stages.  Stage one scores confident detections with the
learned agent/critic utilities.  Stage two lets recently seen agents claim
remaining low-confidence boxes by IoU only, without writing their appearance.
"""

from __future__ import annotations
from dataclasses import fields
import torch
from evoworld.geometry import sample_features, warp_boxes, pair_iou
from evoworld.classes import same_group
from evoworld.models.world import ObjectState, damp_missing
from evoworld.models.life import AgentState, Phase, uncertainty_forecast
from evoworld.models.association import blend_memory, assign, assign_with_null
from evoworld.planning import ComputeBudget, allocate_planning, imagine, pooled_features, select_objects


def policy_ready(model, config):
    mode = config.al_inference_policy
    if mode == "auto":
        return int(model.al.policy_examples) >= config.al_min_examples
    return mode == "on"


def critic_ready(model, config):
    mode = config.al_inference_critic
    if mode == "auto":
        return int(model.al.critic_examples) >= config.al_critic_min_examples
    return mode == "on"


class LifeTracker:
    def __init__(self, model):
        self.model, self.config = model, model.config
        self.reset()

    def reset(self):
        self.state = None
        self.agents = None
        self.memory = self.anchor = self.votes = self.ids = None
        self.pending = []
        self.camera_state = self.previous_camera = self.previous_frame = None
        self.next_id = 1
        self.last_diagnostics = {}
        self.last_backfill = []

    @property
    def classes(self):
        """Track class = confidence-weighted majority of its observed detector classes."""
        if self.votes is None:
            return None
        return self.votes.argmax(-1)

    @torch.no_grad()
    def step(self, detections, dense, camera, frame_id, affine=None):
        if self.previous_frame is not None and frame_id <= self.previous_frame:
            raise ValueError("Frames must be strictly increasing; reset between sequences")
        c, model = self.config, self.model
        dt = camera.new_tensor([1 if self.previous_frame is None else frame_id - self.previous_frame])
        previous = camera if self.previous_camera is None else self.previous_camera
        self.camera_state, learned = model.camera.observe(camera, previous, self.camera_state)
        motion = learned if affine is None else affine.reshape(1, 2, 3).to(camera)
        boxes, confidence, classes = (detections[k] for k in ("boxes", "scores", "classes"))
        # Center features are the only appearance written to memory/world/birth, as in training.
        center = sample_features(dense, boxes[None])[0]
        if self.state is None:
            self.state = model.world.initial(dense, 1, 0)
            self.agents = AgentState.initial(dense, 0)
            self.memory = dense.new_empty(0, c.embedding_dim)
            self.anchor = self.memory.clone()
            self.ids = torch.empty(0, dtype=torch.long, device=dense.device)
            self.votes = dense.new_zeros(0, c.num_classes)
        n, m = len(self.ids), len(boxes)
        high = (confidence >= c.conf_threshold).nonzero().flatten()
        low = (confidence < c.conf_threshold).nonzero().flatten()
        budget = ComputeBudget(c.al_compute_budget)
        matched, output, terminated, recovered, backfill = set(), [], [], [], []
        ready, judged = policy_ready(model, c), critic_ready(model, c)
        observed = torch.zeros(1, n, dtype=torch.bool, device=boxes.device)
        if n:
            predicted, forecast = model.world.predict(self.state, motion, dt)
            agents = self.agents.predict(forecast, dt, c, predicted.box[0])
            inflated = uncertainty_forecast(forecast, agents)
            track_classes = self.classes
            features = center.clone()

            def score(columns, values):
                return model.association.scores(
                    predicted,
                    self.memory,
                    self.anchor,
                    boxes[columns],
                    values[columns],
                    track_classes,
                    classes[columns],
                    inflated,
                    confidence[columns],
                )

            everything = torch.arange(m, device=boxes.device)
            scores = score(everything, features)
            # Statistics see every candidate, matching training where distractors are present.
            statistics = agents.statistics(predicted, forecast, scores, c)
            policies = model.al.agents(predicted.h[0], statistics)
            ambiguous = (statistics[:, 9] < c.uncertain_margin / 5) | (agents.age > 1)
            eligibility = (policies["resource"][:, 0] > c.al_acquisition_cost) if ready else ambiguous
            eligibility &= agents.credit >= c.al_plan_horizon
            selected = allocate_planning(policies["resource"][:, 0], eligibility, c, budget)
            agents.credit = agents.credit.clone()
            agents.credit[selected] -= c.al_plan_horizon
            cache = imagine(
                model, predicted, model.camera.imagine(self.camera_state), selected, c.al_plan_horizon
            )
            summary = cache.summary(self.memory)
            if len(high):
                # ROI rechecks refine scoring only; each pooled detection is computed once.
                priority = policies["resource"][:, 1]
                requests = (priority > c.al_acquisition_cost) if ready else ambiguous
                queried = set()
                for i in priority.argsort(descending=True).tolist():
                    row = scores[i, high]
                    if not requests[i] or agents.credit[i] < 1 or not torch.isfinite(row).any():
                        continue
                    j = int(high[row.argmax()])
                    if j in queried:
                        continue
                    if not budget.spend(1, "roi"):
                        break
                    features[j] = pooled_features(dense, boxes[j : j + 1][None])[0, 0]
                    agents.credit[i] -= 1
                    queried.add(j)
                if queried:
                    scores = score(everything, features)
            adjusted, chosen_gate, null = self._decisions(
                predicted,
                agents,
                statistics,
                policies,
                summary,
                cache,
                scores[:, high],
                boxes[high],
                center[high],
                confidence[high],
                ready,
                judged,
            )
            pairs = [(i, int(high[k]), chosen_gate[i, k]) for i, k in assign_with_null(adjusted, null)]
            # Stage two: recently seen agents keep low-confidence (often occluded) evidence.
            taken = {i for i, _, _ in pairs}
            waiting = [i for i in range(n) if i not in taken and int(agents.age[i]) <= c.low_conf_max_age]
            if waiting and len(low):
                rows = torch.tensor(waiting, device=boxes.device)
                overlap = pair_iou(predicted.box[0, rows], boxes[low])
                compatible = same_group(track_classes[rows], classes[low], c)
                overlap = overlap.masked_fill(~compatible, -1.0)
                for a, k in assign(overlap, c.low_conf_iou):
                    pairs.append((waiting[a], int(low[k]), boxes.new_zeros(())))
            measured, appearance = predicted.box.clone(), self.memory.clone()
            memory = self.memory.clone()
            votes = self.votes * 0.98
            measured_confidence = boxes.new_zeros(n)
            innovation = boxes.new_zeros(n)
            for i, j, gate in pairs:
                matched.add(j)
                observed[0, i] = True
                measured[0, i], appearance[i] = boxes[j], center[j]
                measured_confidence[i] = confidence[j]
                innovation[i] = (boxes[j, :2] - predicted.box[0, i, :2]).norm() / predicted.box[
                    0, i, 2:
                ].mean().clamp_min(0.02)
                memory[i] = blend_memory(self.memory[i], center[j], gate)
                votes[i, classes[j]] += confidence[j]
                if self.agents.phase[i] == Phase.DORMANT:
                    recovered.append(int(self.ids[i]))
            state, _ = model.world.correct(
                predicted, measured, appearance[None], observed, dt, warp_boxes(self.state.box, motion)
            )
            self.state = damp_missing(state, observed, dt, c.dormant_velocity_decay)
            self.memory, self.votes = memory, votes
            was_confirmed = self.agents.hits >= c.al_confirm_hits
            self.agents = agents.observe(
                observed[0],
                measured_confidence,
                measured[0],
                innovation,
                policies["existence"].sigmoid(),
                ready,
                c,
            )
            end = (
                ~observed[0]
                & (self.agents.age >= c.al_exit_grace)
                & ready
                & (self.agents.existence < c.al_existence_threshold)
                & (policies["lifecycle"].softmax(-1)[:, 2] >= c.al_terminate_probability)
            )
            end |= self.agents.age > c.al_max_dormant_age
            end |= (self.agents.hits < c.al_confirm_hits) & (self.agents.age > c.al_tentative_age)
            label = self.classes
            for i, j, _ in pairs:
                row = (int(self.ids[i]), j, boxes[j], float(confidence[j]), int(label[i]))
                if self.agents.hits[i] < c.al_confirm_hits:
                    self.pending[i].append((frame_id, *row[1:]))
                    continue
                if confidence[j] < c.conf_threshold and self.agents.phase[i] != Phase.ACTIVE:
                    continue
                output.append(row)
                if not was_confirmed[i] and c.backfill_tentative:
                    backfill.extend((f, row[0], *rest[1:]) for f, *rest in self.pending[i])
                self.pending[i] = []
            terminated = self.ids[end].tolist()
            keep = ~end
            self.state = select_objects(self.state, keep)
            self.agents = self.agents.select(keep)
            self.memory, self.anchor = self.memory[keep], self.anchor[keep]
            self.ids, self.votes = self.ids[keep], self.votes[keep]
            self.pending = [p for p, k in zip(self.pending, keep.tolist()) if k]
            observed = observed[:, keep]
        births = [
            j for j in high.tolist() if j not in matched and confidence[j] >= c.birth_threshold
        ]
        if births and observed.any():
            # A box overlapping a just-matched agent of the same group is a duplicate, not a birth.
            index = torch.tensor(births, device=boxes.device)
            owners = observed[0].nonzero().flatten()
            overlap = pair_iou(boxes[index], self.state.box[0, owners])
            overlap = overlap.masked_fill(~same_group(classes[index], self.classes[owners], c), 0)
            births = [j for j, o in zip(births, overlap.max(-1).values.tolist()) if o <= c.birth_suppress_iou]
        if births:
            index = torch.tensor(births, device=boxes.device)
            initial = model.world.initial(boxes, 1, len(index))
            born, _ = model.world.correct(
                initial,
                boxes[index][None],
                center[index][None],
                torch.ones(1, len(index), dtype=torch.bool, device=boxes.device),
                dt,
            )
            if ready:
                birth_policy = model.al.agents(born.h[0], model.al.birth_statistics(confidence[index], c))
                accepted = birth_policy["existence"].sigmoid() >= c.al_birth_probability
                index = index[accepted]
                born = select_objects(born, accepted)
                births = [j for j, yes in zip(births, accepted.tolist()) if yes]
            if births:
                self.state = ObjectState(
                    **{
                        f.name: torch.cat((getattr(self.state, f.name), getattr(born, f.name)), 1)
                        for f in fields(born)
                    }
                )
                agents = AgentState.initial(boxes, len(index))
                agents.credit.clamp_max_(c.al_credit_capacity)
                agents.hits.fill_(1)
                agents.lifetime.fill_(1)
                agents.existence = 0.5 + 0.5 * confidence[index]
                if c.al_confirm_hits <= 1:
                    agents.phase.fill_(Phase.ACTIVE)
                self.agents = self.agents.append(agents)
                self.memory = torch.cat((self.memory, center[index]))
                self.anchor = torch.cat((self.anchor, center[index]))
                identity = torch.arange(self.next_id, self.next_id + len(index), device=boxes.device)
                self.next_id += len(index)
                self.ids = torch.cat((self.ids, identity))
                vote = boxes.new_zeros(len(index), c.num_classes)
                vote[torch.arange(len(index)), classes[index]] = confidence[index]
                self.votes = torch.cat((self.votes, vote))
                for i, j in zip(identity.tolist(), births):
                    row = (int(i), j, boxes[j], float(confidence[j]), int(classes[j]))
                    if c.al_confirm_hits <= 1:
                        output.append(row)
                        self.pending.append([])
                    else:
                        self.pending.append([(frame_id, *row[1:])])
        self.previous_frame, self.previous_camera = frame_id, camera
        self.last_backfill = backfill
        self.last_diagnostics = dict(
            agents=len(self.ids),
            dormant=int((self.agents.phase == Phase.DORMANT).sum()),
            recovered_ids=recovered,
            terminated_ids=terminated,
            backfilled=len(backfill),
            compute_tokens=budget.spent,
            forecast_agent_steps=budget.forecast_steps,
            roi_rechecks=budget.roi_rechecks,
            states=[
                dict(
                    id=int(identity),
                    phase=Phase(int(self.agents.phase[k])).name.lower(),
                    existence=float(self.agents.existence[k]),
                    uncertainty=self.agents.variance[k].sqrt().tolist(),
                    age=int(self.agents.age[k]),
                    credit=float(self.agents.credit[k]),
                )
                for k, identity in enumerate(self.ids)
            ],
        )
        return output

    def _decisions(
        self,
        predicted,
        agents,
        statistics,
        policies,
        summary,
        cache,
        scores,
        boxes,
        features,
        confidence,
        ready,
        critic_ready,
    ):
        """Vectorized agent/detection/action scoring; no CNN or physical rerollout per action."""
        model, c = self.model, self.config
        n, m = scores.shape
        margin = statistics[:, 9] * 5
        null = scores.new_full((n,), c.match_threshold)
        if ready:
            phase = policies["lifecycle"].log_softmax(-1)
            null = null + 0.3 * (phase[:, 1] - phase[:, 0]).clamp(-3, 3)
        if critic_ready:
            # Same wait context as training: anchor, age, margin and forecast affinity.
            wait_pair = model.association.context(
                predicted.h[0],
                self.memory,
                self.anchor,
                self.memory,
                predicted.box[0],
                predicted.box[0],
                null.new_zeros(n),
                agents.age,
                margin,
            )
            wait_base = model.al.context(wait_pair, statistics, summary)
            identity = torch.arange(n, device=scores.device)
            wait_affinity = cache.affinities(self.memory, self.memory, null.new_zeros(n), identity)
            effect = model.al.effects(model.al.action_context(wait_base, null.new_zeros(n), wait_affinity, 1))
            null = null - c.al_risk_weight * (effect[:, 0] + c.effect_margin * effect[:, 2])
        if not m:
            return scores, scores.clone(), null
        memory = self.memory[:, None].expand(-1, m, -1)
        measurement = features[None].expand(n, -1, -1)
        pair = model.association.context(
            predicted.h[0, :, None].expand(-1, m, -1),
            memory,
            self.anchor[:, None].expand(-1, m, -1),
            measurement,
            predicted.box[0, :, None].expand(-1, m, -1),
            boxes[None].expand(n, -1, -1),
            confidence[None].expand(n, -1),
            agents.age[:, None].expand(-1, m),
            margin[:, None].expand(-1, m),
        )
        prior = model.association.gate(pair)
        base = model.al.context(
            pair, statistics[:, None].expand(-1, m, -1), summary[:, None].expand(-1, m, -1)
        )
        quality, proposed = model.al.proposal(base, prior)
        proposed = proposed if ready else prior
        identity = torch.arange(n, device=scores.device)[:, None].expand(-1, m)
        gates, utilities = [], []
        for gate in (proposed, torch.zeros_like(proposed)):
            affinity = cache.affinities(memory, measurement, gate, identity)
            utility = scores.clone()
            if ready:
                utility = utility + 0.3 * torch.nn.functional.logsigmoid(quality)
                utility = utility.masked_fill(quality.sigmoid() < c.al_match_probability, -torch.inf)
                # Row-centered: forecast evidence ranks detections for one agent, but does not
                # make planned agents systematically more eager than unplanned ones.
                bonus = affinity[..., 0] * affinity[..., 1]
                utility = utility + c.al_forecast_weight * (bonus - bonus.mean(-1, keepdim=True))
            if critic_ready:
                effect = model.al.effects(model.al.action_context(base, gate, affinity))
                utility = utility - c.al_risk_weight * (effect[..., 0] + c.effect_margin * effect[..., 2])
            gates.append(gate)
            utilities.append(utility)
        utility = torch.stack(utilities)
        choice = utility.argmax(0)
        return (utility.gather(0, choice[None])[0], torch.stack(gates).gather(0, choice[None])[0], null)

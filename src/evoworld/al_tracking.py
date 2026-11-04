"""Per-identity life agents with learned maintenance and bounded acquisition."""

from __future__ import annotations
import torch
from evoworld.geometry import sample_features, warp_boxes
from evoworld.models.world import ObjectState
from evoworld.models.life import AgentState, Phase, uncertainty_forecast
from evoworld.models.association import blend_memory, assign_with_null
from evoworld.planning import ComputeBudget, allocate_planning, imagine, pooled_features, select_objects


class LifeTracker:
    def __init__(self, model):
        self.model, self.config = model, model.config
        self.reset()

    def reset(self):
        self.state = None
        self.agents = None
        self.memory = self.anchor = self.classes = self.ids = None
        self.camera_state = self.previous_camera = self.previous_frame = None
        self.next_id = 1
        self.last_diagnostics = {}

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
        features = sample_features(dense, boxes[None])[0]
        if self.state is None:
            self.state = model.world.initial(dense, 1, 0)
            self.agents = AgentState.initial(dense, 0)
            self.memory = dense.new_empty(0, c.embedding_dim)
            self.anchor = self.memory.clone()
            self.ids = torch.empty(0, dtype=torch.long, device=dense.device)
            self.classes = self.ids.clone()
        n, m = len(self.ids), len(boxes)
        budget = ComputeBudget(c.al_compute_budget)
        matched, output, terminated = set(), [], []
        ready = int(model.al.policy_examples) >= c.al_min_examples
        critic_ready = int(model.al.critic_examples) >= c.al_critic_min_examples
        if n:
            predicted, forecast = model.world.predict(self.state, motion, dt)
            agents = self.agents.predict(forecast, dt, c)
            inflated = uncertainty_forecast(forecast, agents)
            scores = model.association.scores(
                predicted, self.memory, self.anchor, boxes, features, self.classes, classes, inflated
            )
            statistics = agents.statistics(predicted, forecast, scores, c)
            policies = model.al.agents(predicted.h[0], statistics)
            # Learned request value is compared with explicit acquisition cost after warmup.
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
            # Each additional ROI is computed once even if multiple agents request it.
            if m:
                priority = policies["resource"][:, 1]
                order = priority.argsort(descending=True)
                requests = (priority > c.al_acquisition_cost) if ready else ambiguous
                queried = set()
                for i in order.tolist():
                    if not requests[i] or agents.credit[i] < 1 or not torch.isfinite(scores[i]).any():
                        continue
                    j = int(scores[i].argmax())
                    if j in queried:
                        continue
                    if not budget.spend(1, "roi"):
                        break
                    features[j] = pooled_features(dense, boxes[j : j + 1][None])[0, 0]
                    agents.credit[i] -= 1
                    queried.add(j)
                if queried:
                    scores = model.association.scores(
                        predicted, self.memory, self.anchor, boxes, features, self.classes, classes, inflated
                    )
            adjusted, chosen_gate, null = self._decisions(
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
            )
            pairs = assign_with_null(adjusted, null)
            observed = torch.zeros(1, n, dtype=torch.bool, device=boxes.device)
            measured, appearance = predicted.box.clone(), self.memory.clone()
            memory = self.memory.clone()
            measured_confidence = boxes.new_zeros(n)
            innovation = boxes.new_zeros(n)
            recovered = []
            for i, j in pairs:
                matched.add(j)
                observed[0, i] = True
                measured[0, i], appearance[i] = boxes[j], features[j]
                measured_confidence[i] = confidence[j]
                innovation[i] = (boxes[j, :2] - predicted.box[0, i, :2]).norm() / predicted.box[
                    0, i, 2:
                ].mean().clamp_min(0.02)
                memory[i] = blend_memory(self.memory[i], features[j], chosen_gate[i, j])
                if self.agents.phase[i] == Phase.DORMANT:
                    recovered.append(int(self.ids[i]))
            self.state, _ = model.world.correct(
                predicted, measured, appearance[None], observed, dt, warp_boxes(self.state.box, motion)
            )
            self.memory = memory
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
            for i, j in pairs:
                if self.agents.hits[i] >= c.al_confirm_hits and confidence[j] >= c.conf_threshold:
                    output.append((int(self.ids[i]), j, boxes[j], float(confidence[j]), int(classes[j])))
            terminated = self.ids[end].tolist()
            keep = ~end
            self.state = select_objects(self.state, keep)
            self.agents = self.agents.select(keep)
            self.memory, self.anchor = self.memory[keep], self.anchor[keep]
            self.ids, self.classes = self.ids[keep], self.classes[keep]
        else:
            recovered = []
        births = [j for j in range(m) if j not in matched and confidence[j] >= c.birth_threshold]
        if births:
            index = torch.tensor(births, device=boxes.device)
            initial = model.world.initial(boxes, 1, len(index))
            born, _ = model.world.correct(
                initial,
                boxes[index][None],
                features[index][None],
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
                from dataclasses import fields

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
                self.memory = torch.cat((self.memory, features[index]))
                self.anchor = torch.cat((self.anchor, features[index]))
                identity = torch.arange(self.next_id, self.next_id + len(index), device=boxes.device)
                self.next_id += len(index)
                self.ids = torch.cat((self.ids, identity))
                self.classes = torch.cat((self.classes, classes[index]))
                if c.al_confirm_hits <= 1:
                    output.extend(
                        (int(i), j, boxes[j], float(confidence[j]), int(classes[j]))
                        for i, j in zip(identity, births)
                    )
        self.previous_frame, self.previous_camera = frame_id, camera
        self.last_diagnostics = dict(
            agents=len(self.ids),
            dormant=int((self.agents.phase == Phase.DORMANT).sum()),
            recovered_ids=recovered,
            terminated_ids=terminated,
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
        null = scores.new_full((n,), c.match_threshold)
        if ready:
            phase = policies["lifecycle"].log_softmax(-1)
            null = null + 0.3 * (phase[:, 1] - phase[:, 0]).clamp(-3, 3)
        if critic_ready:
            wait_pair = model.association.context(
                predicted.h[0],
                self.memory,
                self.anchor,
                self.memory,
                predicted.box[0],
                predicted.box[0],
                null.new_zeros(n),
                agents.age,
                statistics[:, 9] * 5,
            )
            wait_base = model.al.context(wait_pair, statistics, summary)
            wait_affinity = null.new_zeros(n, 2)
            wait_affinity[:, 1] = summary[:, 0]
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
            statistics[:, 9, None].expand(-1, m) * 5,
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
                utility = utility + c.al_forecast_weight * affinity[..., 0] * affinity[..., 1]
            if critic_ready:
                effect = model.al.effects(model.al.action_context(base, gate, affinity))
                utility = utility - c.al_risk_weight * (effect[..., 0] + c.effect_margin * effect[..., 2])
            gates.append(gate)
            utilities.append(utility)
        utility = torch.stack(utilities)
        choice = utility.argmax(0)
        return (utility.gather(0, choice[None])[0], torch.stack(gates).gather(0, choice[None])[0], null)

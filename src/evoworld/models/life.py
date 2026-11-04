"""Shared policies for per-track self-maintenance, acquisition and finite-time value.

A track is an inference agent, not a controllable physical object. Its state can
change its memory and decisions, but never changes a cached physical forecast.
"""

from __future__ import annotations
from dataclasses import dataclass, fields, replace
from enum import IntEnum
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .world import mlp


class Phase(IntEnum):
    TENTATIVE = 0
    ACTIVE = 1
    DORMANT = 2


@dataclass
class AgentState:
    existence: Tensor
    variance: Tensor
    age: Tensor
    hits: Tensor
    lifetime: Tensor
    error: Tensor
    phase: Tensor
    credit: Tensor

    @classmethod
    def initial(cls, reference, count):
        scalar = reference.new_zeros(count)
        integer = torch.zeros(count, dtype=torch.long, device=reference.device)
        return cls(
            scalar + 0.8,
            reference.new_full((count, 2), 0.0001),
            integer.clone(),
            integer.clone(),
            integer.clone(),
            scalar.clone(),
            integer.clone(),
            scalar + 3.0,
        )

    def select(self, index):
        return AgentState(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def detached(self):
        return AgentState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def append(self, other):
        return AgentState(
            **{f.name: torch.cat((getattr(self, f.name), getattr(other, f.name))) for f in fields(self)}
        )

    def predict(self, forecast, delta, config):
        dt = delta.reshape(-1)[0]
        return replace(
            self,
            variance=(self.variance + forecast["log_sigma"][0, :, :2].exp().square() * dt).clamp_max(1.0),
            age=self.age + dt.long(),
            lifetime=self.lifetime + dt.long(),
            credit=(self.credit + config.al_credit_refill * dt).clamp_max(config.al_credit_capacity),
        )

    def statistics(self, predicted, forecast, scores, config):
        n = len(self.age)
        if scores.shape[1]:
            finite = torch.where(torch.isfinite(scores), scores, scores.new_full((), -20.0))
            top = finite.max(-1).values
            alternatives = torch.cat((finite, finite.new_full((n, 1), config.match_threshold)), -1)
            best = alternatives.topk(min(2, alternatives.shape[1]), -1).values
            margin = best[:, 0] - best[:, -1]
        else:
            top = self.existence.new_full((n,), config.match_threshold)
            margin = torch.zeros_like(top)
        center = predicted.box[0, :, :2]
        inside = ((center >= 0) & (center <= 1)).all(-1).float()
        return torch.stack(
            (
                self.existence,
                self.variance[:, 0].sqrt(),
                self.variance[:, 1].sqrt(),
                self.age.float() / max(1, config.al_max_dormant_age),
                self.hits.float().clamp_max(10) / 10,
                self.lifetime.float().clamp_max(100) / 100,
                self.error,
                (self.phase == Phase.DORMANT).float(),
                top.clamp(-10, 10) / 5,
                margin.clamp(0, 10) / 5,
                forecast["visibility"][0].sigmoid(),
                inside,
                self.credit / config.al_credit_capacity,
            ),
            -1,
        )

    def observe(self, observed, confidence, boxes, innovation, alive_probability, ready, config):
        existence = 0.8 * self.existence + 0.2 * alive_probability if ready else self.existence * 0.98
        existence = torch.where(observed, torch.maximum(existence, 0.5 + 0.5 * confidence), existence)
        hits = self.hits + observed.long()
        phase = torch.where(
            observed,
            torch.where(
                hits >= config.al_confirm_hits,
                torch.full_like(hits, Phase.ACTIVE),
                torch.full_like(hits, Phase.TENTATIVE),
            ),
            torch.full_like(hits, Phase.DORMANT),
        )
        variance = torch.where(observed[:, None], (0.05 * boxes[:, 2:] + 0.002).square(), self.variance)
        error = torch.where(observed, 0.8 * self.error + 0.2 * innovation.clamp(0, 1), self.error)
        return AgentState(
            existence,
            variance,
            torch.where(observed, 0, self.age),
            hits,
            self.lifetime,
            error,
            phase,
            self.credit,
        )


class LifeController(nn.Module):
    statistics_dim = 13
    summary_dim = 4

    @classmethod
    def birth_statistics(cls, confidence, config):
        stats = confidence.new_zeros(len(confidence), cls.statistics_dim)
        stats[:, 0] = 0.5 + 0.5 * confidence
        stats[:, 1:3] = 0.01
        stats[:, 4], stats[:, 5], stats[:, 10:12] = 0.1, 0.01, 1.0
        stats[:, 12] = min(3.0, config.al_credit_capacity) / config.al_credit_capacity
        return stats

    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.agent = mlp(h + self.statistics_dim, h, h)
        self.existence = nn.Linear(h, 1)
        self.lifecycle = nn.Linear(h, 3)  # continue, dormant, terminate
        self.resource = nn.Linear(h, 2)  # value of planning, value of ROI recheck
        base = h + 8 + self.statistics_dim + self.summary_dim
        self.candidate = mlp(base, h, 2)  # match correctness and residual write logit
        self.critic = mlp(base + 5, h, 3)  # two forecast affinities, three action-type flags
        self.horizon = config.probe_horizon
        self.register_buffer("policy_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("critic_examples", torch.tensor(0, dtype=torch.long))
        for head in (self.existence, self.lifecycle, self.resource):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.existence.bias, 2.0)
        with torch.no_grad():
            self.lifecycle.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
            self.resource.bias.fill_(-3.0)
        nn.init.zeros_(self.candidate[-1].weight)
        nn.init.zeros_(self.candidate[-1].bias)

    def agents(self, hidden, statistics):
        encoded = self.agent(torch.cat((hidden, statistics), -1))
        return dict(
            existence=self.existence(encoded).squeeze(-1),
            lifecycle=self.lifecycle(encoded),
            resource=F.softplus(self.resource(encoded)),
        )

    def context(self, pair_context, statistics, summary):
        return torch.cat((pair_context, statistics, summary), -1)

    def proposal(self, context, prior_gate):
        raw = self.candidate(context)
        logit = torch.logit(prior_gate.clamp(0.001, 0.999))
        return raw[..., 0], torch.sigmoid(logit + raw[..., 1])

    def action_context(self, context, gate, affinities, action_type=0):
        context = context.clone()
        # Association's last entry is the proposed memory-write gate.
        context[..., self._gate_index] = gate
        flag = F.one_hot(torch.full_like(gate, action_type, dtype=torch.long), 3).to(context)
        return torch.cat((context, affinities, flag), -1)

    @property
    def _gate_index(self):
        return self.candidate[0].in_features - self.statistics_dim - self.summary_dim - 1

    def effects(self, action_context):
        raw = self.critic(action_context)
        return torch.stack(
            (raw[..., 0].sigmoid() * self.horizon, F.softplus(raw[..., 1]), F.softplus(raw[..., 2]) + 0.01),
            -1,
        )


def uncertainty_forecast(forecast, agents):
    enriched = dict(forecast)
    variance = agents.variance.detach().clamp_min(1e-8)
    enriched["log_sigma"] = torch.cat((0.5 * variance.log()[None], forecast["log_sigma"][..., 2:]), -1)
    enriched["search_variance"] = variance
    return enriched

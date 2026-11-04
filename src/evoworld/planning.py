"""Budgeted acquisition and action-independent, cached physical imagination."""

from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch import Tensor
import torch.nn.functional as F
from evoworld.geometry import sample_features
from evoworld.models.world import ObjectState
from evoworld.models.association import blend_memory


def select_objects(state, indices):
    return ObjectState(**{f.name: getattr(state, f.name)[:, indices] for f in fields(state)})


def pooled_features(dense, boxes):
    """Five real image samples per actual detection; no predicted feature is a measurement."""
    offsets = boxes.new_tensor([[0.0, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.0, -0.25], [0.0, 0.25]])
    centers = boxes[..., None, :2] + boxes[..., None, 2:] * offsets
    points = torch.cat((centers, boxes[..., None, 2:].expand_as(centers)), -1)
    feature = sample_features(dense, points.flatten(1, 2)).reshape(*boxes.shape[:2], 5, dense.shape[1])
    return F.normalize(feature.mean(-2), dim=-1, eps=1e-6)


@dataclass
class ForecastCache:
    indices: Tensor
    boxes: Tensor
    features: Tensor
    visibility: Tensor
    sigma: Tensor
    agent_count: int

    def summary(self, reference):
        result = reference.new_zeros(self.agent_count, 4)
        if len(self.indices):
            # One summary per selected identity, shared by all its action candidates.
            value = torch.stack(
                (
                    self.visibility.mean(0),
                    self.sigma.mean((0, 2)),
                    self.boxes[..., 2:].mean((0, 2)),
                    self.visibility.var(0, unbiased=False),
                ),
                -1,
            )
            result[self.indices] = value
        return result

    def affinities(self, memory, measurement, gate, track_indices):
        """Evaluate a hypothetical memory without modifying physical futures or real memory."""
        out = memory.new_zeros(*gate.shape, 2)
        if not len(self.indices):
            return out
        hypothetical = blend_memory(memory, measurement, gate)
        for local, identity in enumerate(self.indices.tolist()):
            mask = track_indices == identity
            if not mask.any():
                continue
            expected = self.features[:, local].mean(0)
            out[..., 0][mask] = (hypothetical[mask] * expected).sum(-1)
            out[..., 1][mask] = self.visibility[:, local].mean()
        return out


@torch.no_grad()
def imagine(model, state, camera_prior, indices, horizon, delta=None):
    """Selected agents share one physical forecast across all candidate tracker actions.

    Current hidden states already encode the full current neighbor graph. Future
    neighbor interaction is restricted to the selected set, an explicit approximation.
    """
    count = state.box.shape[1]
    if not len(indices) or horizon <= 0:
        empty = state.box.new_empty(0, 0, 4)
        return ForecastCache(indices[:0], empty, empty, empty[..., 0], empty, count)
    current = select_objects(state.detached(), indices)
    dt = state.box.new_ones(1) if delta is None else delta
    boxes, features, visibility, sigma = [], [], [], []
    for _ in range(horizon):
        current, forecast = model.world.predict(current, camera_prior, dt)
        boxes.append(current.box[0])
        features.append(forecast["feature"][0])
        visibility.append(forecast["visibility"][0].sigmoid())
        sigma.append(forecast["log_sigma"][0].exp())
    return ForecastCache(
        indices, torch.stack(boxes), torch.stack(features), torch.stack(visibility), torch.stack(sigma), count
    )


@dataclass
class ComputeBudget:
    """Tokens count actual agent forecast steps and additional detection ROI pools."""

    limit: int
    spent: int = 0
    forecast_steps: int = 0
    roi_rechecks: int = 0

    def spend(self, amount, kind):
        if amount < 0 or self.spent + amount > self.limit:
            return False
        self.spent += amount
        if kind == "forecast":
            self.forecast_steps += amount
        elif kind == "roi":
            self.roi_rechecks += amount
        else:
            raise ValueError(kind)
        return True


def allocate_planning(priority, eligible, config, budget):
    slots = min(config.al_planning_agents, (budget.limit - budget.spent) // config.al_plan_horizon)
    if not slots:
        return torch.empty(0, dtype=torch.long, device=priority.device)
    index = eligible.nonzero().flatten()
    index = index[priority[index].argsort(descending=True)[:slots]]
    assert budget.spend(len(index) * config.al_plan_horizon, "forecast")
    return index

"""Association, measurement-only memory writes, and learned intervention effects."""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from evoworld.geometry import pair_iou
from .world import mlp


def blend_memory(memory: Tensor, measurement: Tensor, gate: Tensor) -> Tensor:
    """Only call with an actual observed appearance, never imagined appearance."""
    return F.normalize((1 - gate[..., None]) * memory + gate[..., None] * measurement, dim=-1, eps=1e-6)


def assign(scores: Tensor, threshold: float, forbidden: tuple[int, int] | None = None):
    """One-to-one assignment with an independent unmatched option for every row."""
    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    values = scores.detach().float().cpu().numpy().copy()
    if forbidden is not None:
        values[forbidden] = -np.inf
    values = np.concatenate((values, np.full((n, n), threshold)), axis=1)
    rows, cols = linear_sum_assignment(-values)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if j < m and values[i, j] > threshold]


SCORE_WEIGHTS = (2.0, 0.5, 1.5, 0.3, 0.1, 0.2)  # cosine, anchor, IoU, offset, innovation, visibility


class Association(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Pair features: cosine, anchor cosine, relative offset(2), log size(2), confidence.
        self.pair = mlp(config.hidden_dim + 7, config.hidden_dim, 1)
        self.write = mlp(config.hidden_dim + 8, config.hidden_dim, 1)
        self.effect = mlp(config.hidden_dim + 8, config.hidden_dim, 3)
        # Hand-set evidence weights become learnable scalars initialized at the old values.
        self.score_weights = nn.Parameter(torch.tensor(SCORE_WEIGHTS))
        self.horizon = config.probe_horizon
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)
        nn.init.zeros_(self.write[-1].weight)
        nn.init.constant_(self.write[-1].bias, -2.0)
        self.register_buffer("effect_examples", torch.tensor(0, dtype=torch.long))

    def scores(
        self,
        state,
        memory,
        anchor,
        boxes,
        features,
        classes,
        detection_classes,
        forecast=None,
        confidence=None,
    ):
        """Pair scores; -inf marks impossible pairs (class group or position gate)."""
        from evoworld.classes import class_compatibility

        n, m = memory.shape[0], features.shape[0]
        if n == 0 or m == 0:
            return memory.new_empty(n, m)
        w = self.score_weights.float()
        cosine = memory @ features.T
        anchor_cosine = anchor @ features.T
        diff = boxes[None] - state.box[0, :, None]
        size = state.box[0, :, None, 2:].clamp_min(1e-4)
        scale = size.clamp_min(0.02)
        if forecast is not None and "search_variance" in forecast:
            scale = torch.maximum(scale, 2 * forecast["search_variance"][:, None].sqrt())
        relative = diff[..., :2] / scale
        log_size = (boxes[None, :, 2:].clamp_min(1e-4) / size).log()
        iou = pair_iou(state.box[0], boxes)
        if confidence is None:
            confidence = features.new_ones(m)
        pair_input = torch.cat(
            (
                cosine[..., None],
                anchor_cosine[..., None],
                relative,
                log_size,
                confidence.to(cosine)[None, :, None].expand(n, -1, -1),
            ),
            -1,
        )
        h = state.h[0, :, None].expand(-1, m, -1)
        learned = self.pair(torch.cat((h, pair_input), -1)).squeeze(-1)
        distance = relative.square().sum(-1)
        score = (
            w[0] * cosine
            + w[1] * anchor_cosine
            + w[2] * iou
            - w[3] * distance.clamp_max(50)
            + learned
        )
        if forecast is not None:
            # Noise floor scales with object size: large boxes jitter more in absolute terms.
            floor = (0.25 * size[..., :2]).clamp_min(0.004)
            sigma = torch.maximum(forecast["log_sigma"][0, :, None, :2].exp(), floor)
            innovation = (boxes[None, :, :2] - state.box[0, :, None, :2]) / sigma
            score = score - w[4] * innovation.square().sum(-1).clamp_max(50)
            score = score + w[5] * F.logsigmoid(forecast["visibility"][0])[:, None]
        allowed, penalty = class_compatibility(classes, detection_classes, self.config)
        score = score - penalty.to(score)
        if self.config.position_gate > 0:
            allowed = allowed & (distance.detach() <= self.config.position_gate**2)
        return score.masked_fill(~allowed, -torch.inf)

    def context(
        self, h, memory, anchor, measurement, predicted_box, measured_box, confidence, age, margin, gate=None
    ):
        # All context is available at the current frame. No future targets enter here.
        cosine = (memory * measurement).sum(-1)
        anchor_cosine = (anchor * measurement).sum(-1)
        innovation = (
            (measured_box[..., :2] - predicted_box[..., :2]) / predicted_box[..., 2:].clamp_min(0.02)
        ).norm(dim=-1).clamp_max(10) / 10
        scale = (
            measured_box[..., 2:].clamp_min(1e-4) / predicted_box[..., 2:].clamp_min(1e-4)
        ).log().abs().mean(-1).clamp_max(5) / 5
        gate = torch.zeros_like(cosine) if gate is None else gate
        extra = torch.stack(
            (
                cosine,
                anchor_cosine,
                innovation,
                scale,
                confidence,
                age.float().clamp_max(30) / 30,
                margin.clamp(-5, 5) / 5,
                gate,
            ),
            -1,
        )
        return torch.cat((h, extra), -1)

    def gate(self, context):
        return torch.sigmoid(self.write(context).squeeze(-1))

    def effects(self, context):
        raw = self.effect(context)
        return torch.stack(
            (
                torch.sigmoid(raw[..., 0]) * self.horizon,
                F.softplus(raw[..., 1]),
                F.softplus(raw[..., 2]) + 0.01,
            ),
            -1,
        )


def assign_with_null(scores: Tensor, null_scores: Tensor):
    """Global one-to-one arbitration with an agent-specific wait value."""
    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    value = scores.detach().float().cpu().numpy()
    null = null_scores.detach().float().cpu().numpy()
    full = np.concatenate((value, np.repeat(null[:, None], n, axis=1)), axis=1)
    rows, cols = linear_sum_assignment(-full)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if j < m and value[i, j] > null[i]]

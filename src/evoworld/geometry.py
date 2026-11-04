"""Normalized image coordinates throughout temporal modeling and association."""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    return torch.cat(
        ((boxes[..., :2] + boxes[..., 2:]) / 2, (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-5)), -1
    )


def pair_iou(a: Tensor, b: Tensor) -> Tensor:
    aa, bb = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    lt = torch.maximum(aa[:, None, :2], bb[None, :, :2])
    rb = torch.minimum(aa[:, None, 2:], bb[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    union = a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter
    return inter / union.clamp_min(1e-8)


def identity_affine(reference: Tensor, *shape: int) -> Tensor:
    return reference.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).expand(*shape, 2, 3).clone()


def warp_boxes(boxes: Tensor, affine: Tensor) -> Tensor:
    """Axis-aligned envelope of an affine-transformed normalized box."""
    matrix, shift = affine[..., :2], affine[..., 2]
    while matrix.ndim < boxes.ndim + 1:
        matrix = matrix.unsqueeze(-3)
        shift = shift.unsqueeze(-2)
    center = (matrix @ boxes[..., :2, None]).squeeze(-1) + shift
    size = (matrix.abs() @ boxes[..., 2:, None]).squeeze(-1)
    return torch.cat((center, size.clamp(1e-5, 2.0)), -1)


def sample_features(dense: Tensor, boxes: Tensor) -> Tensor:
    """Pixel-center contract matches grid_sample(align_corners=False)."""
    if not boxes.shape[1]:
        return dense.new_empty(dense.shape[0], 0, dense.shape[1])
    grid = (2 * boxes[..., :2] - 1).unsqueeze(1)
    values = F.grid_sample(dense, grid, align_corners=False, padding_mode="border")
    return F.normalize(values.squeeze(2).transpose(1, 2), dim=-1, eps=1e-6)

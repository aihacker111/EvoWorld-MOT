"""Confusable-class groups shared by training, association and inference.

Detectors on aerial video often flip between visually adjacent categories
(car/van, pedestrian/people).  A hard class mask turns every such flip into a
lost identity, so association allows matches inside a group with a soft
penalty, forbids matches across groups, and tracks vote for their class.
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from evoworld.geometry import cxcywh_to_xyxy


_TABLES: dict = {}


def group_table(config, device=None) -> Tensor:
    """Cached per (groups, classes, device): association calls this many times per frame."""
    key = (tuple(tuple(int(i) for i in g) for g in config.class_groups), config.num_classes, str(device))
    table = _TABLES.get(key)
    if table is None:
        table = _TABLES[key] = torch.tensor(config.class_group_table(), dtype=torch.long, device=device)
    return table


def class_compatibility(track_classes: Tensor, detection_classes: Tensor, config):
    """Return (allowed[n, m], penalty[n, m]) for track/detection class pairs."""
    table = group_table(config, track_classes.device)
    same = track_classes[:, None] == detection_classes[None]
    group = table[track_classes][:, None] == table[detection_classes][None]
    penalty = (~same & group).float() * config.class_mismatch_penalty
    return group, penalty


def same_group(a: Tensor, b: Tensor, config) -> Tensor:
    table = group_table(config, a.device)
    return table[a][:, None] == table[b][None]


def group_nms(boxes: Tensor, scores: Tensor, classes: Tensor, config) -> Tensor:
    """Greedy NMS across classes of one confusable group; returns kept indices (score order).

    Ultralytics NMS is class-aware, so a car box and a van box on the same object both
    survive; each would otherwise become a separate identity.
    """
    if len(boxes) <= 1 or config.group_nms_iou >= 1:
        return torch.arange(len(boxes), device=boxes.device)
    table = group_table(config, boxes.device)
    corners = cxcywh_to_xyxy(boxes.float()).cpu().numpy()
    groups = table[classes].cpu().numpy()
    order = scores.float().cpu().numpy().argsort()[::-1]
    area = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for index in order:
        if suppressed[index]:
            continue
        keep.append(index)
        lt = np.maximum(corners[index, :2], corners[:, :2])
        rb = np.minimum(corners[index, 2:], corners[:, 2:])
        inter = np.clip(rb - lt, 0, None).prod(-1)
        iou = inter / np.maximum(area[index] + area - inter, 1e-12)
        suppressed |= (iou > config.group_nms_iou) & (groups == groups[index])
    return torch.as_tensor(np.array(keep, dtype=np.int64), device=boxes.device)

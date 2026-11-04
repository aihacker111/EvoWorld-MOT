"""Robust background affine pseudo-labels. No metric 3D/IMU assumptions."""

from __future__ import annotations
import cv2
import numpy as np


def estimate_affine(
    previous: np.ndarray, current: np.ndarray, previous_boxes: np.ndarray | None = None
) -> tuple[np.ndarray, bool]:
    height, width = previous.shape[:2]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    mask = np.full((height, width), 255, np.uint8)
    if previous_boxes is not None:
        for cx, cy, bw, bh in previous_boxes:
            x1, y1 = int((cx - bw * 0.65) * width), int((cy - bh * 0.65) * height)
            x2, y2 = int((cx + bw * 0.65) * width), int((cy + bh * 0.65) * height)
            cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)), 0, -1)
    gray0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray0, 200, 0.01, 5, mask=mask)
    if points is None or len(points) < 8:
        return identity, False
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None)
    if next_points is None:
        return identity, False
    keep = status.reshape(-1).astype(bool)
    p, q = points[keep].reshape(-1, 2), next_points[keep].reshape(-1, 2)
    if len(p) < 8:
        return identity, False
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers.mean() < 0.5 or not np.isfinite(matrix).all():
        return identity, False
    scale = float(np.linalg.det(matrix[:, :2]))
    if not 0.5 < scale < 2.0:
        return identity, False
    pixel = np.eye(3)
    pixel[:2] = matrix
    canvas = np.diag([width, height, 1.0])
    normalized = np.linalg.inv(canvas) @ pixel @ canvas
    return normalized[:2].astype(np.float32), True

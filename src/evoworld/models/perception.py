"""Strict adapter for the supplied Falcon baseline; no Falcon package dependency."""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import yaml
from evoworld.config import Config, resolve, sha256
from evoworld.geometry import xyxy_to_cxcywh
from evoworld.classes import group_nms
from .reid import StandardJDEReID


@dataclass
class FrameFeatures:
    raw: object
    appearance: Tensor
    camera: Tensor


class FalconPerception(nn.Module):
    def __init__(self, config: Config, initialize: bool = True):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG_DICT

        self.config = config
        architecture = yaml.safe_load(resolve(config.detector_yaml).read_text())
        architecture.update(nc=config.num_classes, scale=config.detector_scale)
        self.detector = DetectionModel(cfg=architecture, nc=config.num_classes, verbose=False)
        args = dict(DEFAULT_CFG_DICT)
        self.detector.args = SimpleNamespace(**args)
        head = self.detector.model[-1]
        self.p2_source = head.f[0]
        channels = head.cv2[0][0].conv.in_channels
        self.reid = StandardJDEReID(channels, config.embedding_dim)
        self.load_report = self.load_falcon(resolve(config.detector_checkpoint)) if initialize else {}
        self._criterion = None

    def load_falcon(self, path: Path) -> dict:
        # This user-supplied checkpoint is a trusted local training artifact.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state, dict):
            raise ValueError("Expected a tensor state_dict in the Falcon checkpoint")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        canonical = {k.removeprefix("yolo."): v for k, v in state.items() if k.startswith("yolo.")}
        if not canonical:
            canonical = {k: v for k, v in state.items() if k.startswith("model.")}
        expected = self.detector.state_dict()
        missing = sorted(set(expected) - set(canonical))
        extra = sorted(set(canonical) - set(expected))
        wrong = [k for k in expected.keys() & canonical.keys() if expected[k].shape != canonical[k].shape]
        if missing or extra or wrong:
            raise ValueError(
                f"Detector checkpoint incompatible: missing={missing[:5]}, extra={extra[:5]}, shape={wrong[:5]}"
            )
        self.detector.load_state_dict(canonical, strict=True)
        # Falcon stored aliases of Detect/backbone modules; refuse contradictory aliases.
        last = len(self.detector.model) - 1
        for key, value in state.items():
            alias = None
            if key.startswith("detect_head."):
                alias = f"model.{last}." + key.removeprefix("detect_head.")
            elif key.startswith("backbone_neck."):
                alias = "model." + key.removeprefix("backbone_neck.")
            if alias in canonical and not torch.equal(value, canonical[alias]):
                raise ValueError(f"Checkpoint has contradictory aliases: {key}")
        return {
            "path": str(path),
            "sha256": sha256(path),
            "loaded_tensors": len(expected),
            "epoch": checkpoint.get("epoch"),
            "classes": self.config.num_classes,
            "reid_initialized_from_checkpoint": False,
        }

    def forward(self, images: Tensor) -> FrameFeatures:
        saved, x, p2 = [], images, None
        for layer in self.detector.model[:-1]:
            if layer.f != -1:
                inp = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else saved[j] for j in layer.f]
                )
            else:
                inp = x
            x = layer(inp)
            saved.append(x if layer.i in self.detector.save else None)
            if layer.i == self.p2_source:
                p2 = x
        if p2 is None:
            raise RuntimeError("Detector did not expose the configured P2 feature")
        head = self.detector.model[-1]
        features = [x if j == -1 else saved[j] for j in head.f]
        raw = head(features)
        appearance, _ = self.reid(p2)
        # Compact spatial evidence, intentionally independent of identity memory.
        camera = F.adaptive_avg_pool2d(p2.float().mean(1, keepdim=True), (4, 4)).flatten(1)
        return FrameFeatures(raw, appearance, camera)

    def detection_loss(
        self, raw: object, images: Tensor, boxes: Tensor, labels: Tensor, present: Tensor
    ) -> Tensor:
        if self._criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss

            self._criterion = v8DetectionLoss(self.detector)
        bi, oi = present.nonzero(as_tuple=True)
        target = {
            "img": images,
            "batch_idx": bi,
            "cls": labels[bi, oi, None].float(),
            "bboxes": boxes[bi, oi],
        }
        loss, _ = self._criterion(raw, target)
        return loss.sum() / images.shape[0]

    @torch.no_grad()
    def detections(self, result: FrameFeatures) -> list[dict[str, Tensor]]:
        if self.training:
            raise RuntimeError("Detection decode requires evaluation mode")
        prediction = result.raw[0] if isinstance(result.raw, (tuple, list)) else result.raw
        return self._decode(prediction)

    @torch.no_grad()
    def training_detections(self, raw) -> list[dict[str, Tensor]]:
        # Reuse the already computed feature pyramid; no extra image forward pass.
        if not self.training:
            raise RuntimeError("training_detections expects training-mode head outputs")
        return self._decode(self.detector.model[-1]._inference(raw))

    def _decode(self, prediction):
        from ultralytics.utils.nms import non_max_suppression

        prediction = prediction.float()
        decoded = non_max_suppression(
            prediction,
            conf_thres=self.config.low_threshold,
            iou_thres=self.config.nms_iou,
            max_det=self.config.max_detections,
            nc=self.config.num_classes,
        )
        scale = prediction.new_tensor([self.config.width, self.config.height] * 2)
        output = []
        for row in decoded:
            corners = (row[:, :4] / scale).clamp(0, 1)
            valid = (corners[:, 2:] > corners[:, :2]).all(-1) & torch.isfinite(row).all(-1)
            boxes, scores, classes = xyxy_to_cxcywh(corners[valid]), row[valid, 4], row[valid, 5].long()
            # Same contract in training proposals and inference: one box per object even when
            # the detector fires on two confusable classes.
            keep = group_nms(boxes, scores, classes, self.config)
            output.append({"boxes": boxes[keep], "scores": scores[keep], "classes": classes[keep]})
        return output

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained detector BN running statistics fixed for tiny clip batches.
        # Affine parameters and every convolution remain trainable.
        for module in self.detector.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


class TinyPerception(nn.Module):
    """Explicit synthetic-test backend; never a substitute for the supplied detector."""

    def __init__(self, config: Config, initialize: bool = True):
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU()
        )
        self.reid = nn.Conv2d(32, config.embedding_dim, 1)
        self.objectness = nn.Conv2d(32, 1, 1)
        self.load_report = {"backend": "tiny", "synthetic_only": True}

    def forward(self, images: Tensor) -> FrameFeatures:
        x = self.backbone(images)
        return FrameFeatures(
            self.objectness(x),
            self.reid(x),
            F.adaptive_avg_pool2d(x.mean(1, keepdim=True), (4, 4)).flatten(1),
        )

    def detection_loss(self, raw, images, boxes, labels, present):
        target = torch.zeros_like(raw)
        for b, o in present.nonzero().tolist():
            x, y = boxes[b, o, :2]
            target[
                b,
                0,
                min(int(y * raw.shape[-2]), raw.shape[-2] - 1),
                min(int(x * raw.shape[-1]), raw.shape[-1] - 1),
            ] = 1
        return F.binary_cross_entropy_with_logits(raw, target)

    def detections(self, result):
        raise RuntimeError("tiny backend has no real detection/inference capability")


def build_perception(config: Config, initialize: bool = True) -> nn.Module:
    return (FalconPerception if config.backend == "falcon" else TinyPerception)(config, initialize)

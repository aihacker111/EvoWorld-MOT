"""Class-aware sequence evaluation; export standard MOT files and class sidecars."""

from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evoworld.camera import estimate_affine
from evoworld.config import resolve, atomic_json
from evoworld.data import CocoVideo, prepare_image
from evoworld.geometry import pair_iou, cxcywh_to_xyxy
from evoworld.tracking import EcologyTracker


class IdentityMetrics:
    """IoU=.5 CLEAR matching and global potential-match IDF1, per sequence."""

    def __init__(self):
        self.gt_total = self.pred_total = self.tp = self.switches = self.gt_frames = 0
        self.previous = {}
        self.potential = defaultdict(int)
        self.gt_ids, self.pred_ids = set(), set()

    def update(self, gt_ids, gt_boxes, gt_classes, pred_ids, pred_boxes, pred_classes):
        self.gt_total += len(gt_ids)
        self.pred_total += len(pred_ids)
        self.gt_frames += 1
        self.gt_ids.update(gt_ids)
        self.pred_ids.update(pred_ids)
        similarity = pair_iou(gt_boxes, pred_boxes).cpu().numpy()
        if len(gt_ids) and len(pred_ids):
            similarity *= np.asarray(gt_classes)[:, None] == np.asarray(pred_classes)[None]
        compatible = similarity >= 0.5
        for i, j in np.argwhere(compatible):
            self.potential[(gt_ids[i], pred_ids[j])] += 1
        pairs, used_g, used_p = [], set(), set()
        # Preserve a previous compatible correspondence before solving remaining pairs.
        for i, identity in enumerate(gt_ids):
            old = self.previous.get(identity)
            if old in pred_ids:
                j = pred_ids.index(old)
                if compatible[i, j]:
                    pairs.append((i, j))
                    used_g.add(i)
                    used_p.add(j)
        remaining = similarity.copy()
        for i in used_g:
            remaining[i] = 0
        for j in used_p:
            remaining[:, j] = 0
        if remaining.size:
            row, col = linear_sum_assignment(-remaining)
            pairs += [(int(i), int(j)) for i, j in zip(row, col) if remaining[i, j] >= 0.5]
        for i, j in pairs:
            identity, prediction = gt_ids[i], pred_ids[j]
            if identity in self.previous and self.previous[identity] != prediction:
                self.switches += 1
            self.previous[identity] = prediction
        self.tp += len(pairs)
        return similarity

    def counts(self):
        gt, pred = sorted(self.gt_ids), sorted(self.pred_ids)
        matrix = np.array([[self.potential[(g, p)] for p in pred] for g in gt], dtype=np.float64)
        idtp = 0
        if len(gt) and len(pred):
            rows, cols = linear_sum_assignment(-matrix)
            idtp = int(matrix[rows, cols].sum())
        return dict(
            gt=self.gt_total,
            predictions=self.pred_total,
            tp=self.tp,
            idtp=idtp,
            id_switches=self.switches,
            fp=self.pred_total - self.tp,
            fn=self.gt_total - self.tp,
        )


def summarize(counts):
    result = {
        key: sum(c[key] for c in counts)
        for key in ("gt", "predictions", "tp", "idtp", "id_switches", "fp", "fn")
    }
    result["idf1"] = 2 * result["idtp"] / max(1, result["gt"] + result["predictions"])
    result["mota"] = 1 - (result["fn"] + result["fp"] + result["id_switches"]) / max(1, result["gt"])
    result["precision"] = result["tp"] / max(1, result["predictions"])
    result["recall"] = result["tp"] / max(1, result["gt"])
    return result


def ignore_unmatched(pred_boxes, gt_boxes, gt_classes, pred_classes, ignored, frame):
    """Suppress predictions predominantly inside explicit ignore regions, preserving valid matches."""
    keep = torch.ones(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)
    if not ignored or not len(pred_boxes):
        return keep
    overlap = pair_iou(gt_boxes, pred_boxes)
    if overlap.numel():
        overlap *= gt_classes[:, None] == pred_classes[None]
    valid_match = (overlap >= 0.5).any(0) if len(gt_boxes) else ~keep
    corners = cxcywh_to_xyxy(pred_boxes)
    for annotation in ignored:
        x, y, w, h = annotation["bbox"]
        region = pred_boxes.new_tensor(
            [x / frame["width"], y / frame["height"], (x + w) / frame["width"], (y + h) / frame["height"]]
        )
        inter = (
            (torch.minimum(corners[:, 2:], region[2:]) - torch.maximum(corners[:, :2], region[:2]))
            .clamp_min(0)
            .prod(-1)
        )
        ioa = inter / pred_boxes[:, 2:].prod(-1).clamp_min(1e-8)
        keep &= (ioa <= 0.5) | valid_match
    return keep


def backfill_delay(config):
    """Frames a tentative row can arrive late (0 when backfill is off)."""
    if config.tracker_mode != "al" or not config.backfill_tentative:
        return 0
    return config.al_tentative_age + config.al_confirm_hits


@torch.no_grad()
def evaluate(model, annotations, images, out, device="cpu", hota=False, *, max_frames=None, progress=None):
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if model.config.backend != "falcon":
        raise ValueError("Benchmark evaluation requires the real Falcon detector")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    video = CocoVideo(annotations, images, model.config)
    available_frames = sum(len(frames) for frames in video.sequences.values())
    total_frames = min(available_frames, max_frames) if max_frames is not None else available_frames
    processed = 0

    def report(sequence):
        if progress is not None:
            progress(dict(sequence=sequence, frames=processed, total_frames=total_frames))

    payload = json.loads(resolve(str(annotations)).read_text())
    ignored = defaultdict(list)
    for a in payload.get("annotations", []):
        if a.get("ignore", 0) or a.get("iscrowd", 0) or a["category_id"] == 0:
            ignored[a["image_id"]].append(a)
    counts, sequence_results, hota_results = [], {}, {}
    for seq, frames in sorted(video.sequences.items()):
        if processed >= total_frames:
            break
        frames = frames[: total_frames - processed]
        report(seq)
        tracker = EcologyTracker(model)
        metrics = IdentityMetrics()
        previous, previous_boxes = None, np.empty((0, 4), np.float32)
        lines, sidecar, sequence_data = [], [], []
        # Confirmed tentative rows arrive a few frames late; score frames after that delay.
        delay = backfill_delay(model.config)
        buffer = {}

        def flush(frame_record):
            frame, gt_ids, gt_classes, gt_boxes, rows = frame_record
            pred_boxes = torch.stack([r[2] for r in rows]) if rows else gt_boxes.new_empty(0, 4)
            pred_classes = torch.tensor([r[4] for r in rows], device=gt_boxes.device, dtype=torch.long)
            keep = ignore_unmatched(
                pred_boxes,
                gt_boxes,
                torch.tensor(gt_classes, device=gt_boxes.device, dtype=torch.long),
                pred_classes,
                ignored[frame["id"]],
                frame,
            )
            selected = [r for r, ok in zip(rows, keep) if ok]
            pred_boxes, pred_classes = pred_boxes[keep], pred_classes[keep]
            pred_ids = [r[0] for r in selected]
            similarity = metrics.update(gt_ids, gt_boxes, gt_classes, pred_ids, pred_boxes, pred_classes.tolist())
            sequence_data.append((gt_ids, pred_ids, similarity))
            for identity, _, box, confidence, cls in rows:
                x, y, w, h = box.cpu().tolist()
                x, y, w, h = (
                    (x - w / 2) * frame["width"],
                    (y - h / 2) * frame["height"],
                    w * frame["width"],
                    h * frame["height"],
                )
                lines.append(
                    f"{frame['frame_id']},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{confidence:.5f},-1,-1,-1\n"
                )
                sidecar.append(
                    dict(frame=frame["frame_id"], id=identity, category_id=model.config.category_ids[cls])
                )

        for frame in frames:
            image, annotations_frame = video.read(frame)
            tensor, resized = prepare_image(image, model.config)
            tensor = tensor[None].to(device)
            features = model.perception(tensor)
            detection = model.perception.detections(features)[0]
            affine = None
            if previous is not None:
                transform, valid = estimate_affine(previous, resized, previous_boxes)
                if valid:
                    affine = torch.from_numpy(transform).to(device)
            rows = tracker.step(detection, features.appearance, features.camera, frame["frame_id"], affine)
            gt = video.boxes(frame, annotations_frame)
            gt_ids, gt_classes = [r[0] for r in gt], [r[1] for r in gt]
            gt_boxes = tensor.new_tensor(np.array([r[2] for r in gt], np.float32).reshape(-1, 4))
            buffer[frame["frame_id"]] = (frame, gt_ids, gt_classes, gt_boxes, list(rows))
            for frame_id, identity, box, confidence, cls in getattr(tracker, "last_backfill", []):
                if frame_id in buffer:
                    buffer[frame_id][4].append((identity, -1, box, confidence, cls))
            for frame_id in sorted(buffer):
                if frame_id <= frame["frame_id"] - delay:
                    flush(buffer.pop(frame_id))
            previous, previous_boxes = resized, detection["boxes"].cpu().numpy()
            processed += 1
            if processed % 50 == 0:
                report(seq)
        for frame_id in sorted(buffer):
            flush(buffer.pop(frame_id))
        # No path traversal through user-provided sequence names.
        safe_seq = seq.replace("/", "_").replace("\\", "_").replace("..", "_")
        (out / f"{safe_seq}.txt").write_text("".join(lines))
        (out / f"{safe_seq}.classes.json").write_text(json.dumps(sidecar))
        result = metrics.counts()
        sequence_results[seq] = summarize([result])
        counts.append(result)
        report(seq)
        if hota:
            try:
                from trackeval.metrics import HOTA
            except ImportError as error:
                raise RuntimeError("Install TrackEval to request HOTA: pip install trackeval") from error
            gmap, pmap = (
                {g: i for i, g in enumerate(sorted(metrics.gt_ids))},
                {p: i for i, p in enumerate(sorted(metrics.pred_ids))},
            )
            data = dict(
                num_gt_ids=len(gmap),
                num_tracker_ids=len(pmap),
                num_gt_dets=result["gt"],
                num_tracker_dets=result["predictions"],
                num_timesteps=len(frames),
                gt_ids=[np.array([gmap[g] for g in row[0]], dtype=int) for row in sequence_data],
                tracker_ids=[np.array([pmap[p] for p in row[1]], dtype=int) for row in sequence_data],
                similarity_scores=[row[2] for row in sequence_data],
            )
            hota_results[seq] = HOTA().eval_sequence(data)
    result = summarize(counts)
    result.update(
        sequences=sequence_results,
        category_ids=model.config.category_ids,
        iou_threshold=0.5,
        protocol="Class-aware COCO-MOT; explicit ignore regions; not official VisDrone preprocessing",
        frames=processed,
        available_frames=available_frames,
        max_frames=max_frames,
        truncated=processed < available_frames,
    )
    if hota:
        from trackeval.metrics import HOTA

        combined = HOTA().combine_sequences(hota_results)
        result["hota"] = float(combined["HOTA"].mean())
        result["assa"] = float(combined["AssA"].mean())
        result["deta"] = float(combined["DetA"].mean())
    atomic_json(out / "metrics.json", result)
    return result

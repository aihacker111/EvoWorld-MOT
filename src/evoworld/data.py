"""Ordered COCO-MOT clips, strict sequence identity, and one preprocessing contract."""

from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from evoworld.camera import estimate_affine
from evoworld.config import Config, resolve


def prepare_image(image: np.ndarray, config: Config) -> tuple[torch.Tensor, np.ndarray]:
    resized = cv2.resize(image, (config.width, config.height), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    if config.preprocess == "imagenet_rgb":
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))), resized


class CocoVideo:
    def __init__(self, annotations: str | Path, images: str | Path, config: Config):
        self.path, self.root, self.config = resolve(str(annotations)), resolve(str(images)), config
        payload = json.loads(self.path.read_text())
        self.categories = {cat: i for i, cat in enumerate(config.category_ids)}
        self.annotations = defaultdict(list)
        self.ignored = defaultdict(list)
        for annotation in payload.get("annotations", []):
            if (
                annotation.get("iscrowd", 0)
                or annotation.get("ignore", 0)
                or annotation["category_id"] not in self.categories
            ):
                self.ignored[annotation["image_id"]].append(annotation)
                continue
            if "track_id" not in annotation or annotation["track_id"] < 0:
                raise ValueError(f"MOT requires nonnegative track_id; annotation={annotation.get('id')}")
            self.annotations[annotation["image_id"]].append(annotation)
        self.sequences = defaultdict(list)
        image_ids = [frame["id"] for frame in payload["images"]]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate COCO image ID")
        if set(self.annotations) - set(image_ids):
            raise ValueError("Annotations reference missing images")
        for frame in payload["images"]:
            if "seq_id" not in frame or "frame_id" not in frame:
                raise ValueError("Every image needs seq_id and frame_id; unordered detection COCO is not MOT")
            self.sequences[str(frame["seq_id"])].append(frame)
        if not self.sequences:
            raise ValueError("Dataset has no frames")
        for seq, frames in self.sequences.items():
            frames.sort(key=lambda item: item["frame_id"])
            if len({f["frame_id"] for f in frames}) != len(frames):
                raise ValueError(f"Duplicate frame_id in sequence {seq}")
        identities = sorted(
            {
                (seq, int(a["category_id"]), int(a["track_id"]))
                for seq, frames in self.sequences.items()
                for f in frames
                for a in self.annotations[f["id"]]
            }
        )
        self.identity_index = {key: i for i, key in enumerate(identities)}
        self.lifespans = {}
        for seq, frames in self.sequences.items():
            for frame in frames:
                for annotation in self.annotations[frame["id"]]:
                    key = (seq, int(annotation["category_id"]), int(annotation["track_id"]))
                    identity = self.identity_index[key]
                    if identity not in self.lifespans:
                        self.lifespans[identity] = [frame["frame_id"], frame["frame_id"]]
                    self.lifespans[identity][1] = frame["frame_id"]

    def read(self, frame: dict) -> tuple[np.ndarray, list[dict]]:
        path = self.root / frame["file_name"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read frame: {path}")
        if image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError(f"Annotation/image dimensions disagree: {path}")
        return image, self.annotations[frame["id"]]

    def boxes(self, frame: dict, annotations: list[dict]) -> list[tuple]:
        rows = []
        for annotation in annotations:
            x, y, w, h = map(float, annotation["bbox"])
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(frame["width"]), x + w), min(float(frame["height"]), y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            box = np.array(
                [
                    (x0 + x1) / (2 * frame["width"]),
                    (y0 + y1) / (2 * frame["height"]),
                    (x1 - x0) / frame["width"],
                    (y1 - y0) / frame["height"],
                ],
                np.float32,
            )
            key = (str(frame["seq_id"]), int(annotation["category_id"]), int(annotation["track_id"]))
            rows.append((self.identity_index[key], self.categories[annotation["category_id"]], box))
        ids = [r[0] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate identity in frame {frame['id']}")
        return rows


class VideoClips(Dataset):
    def __init__(self, config: Config, training: bool = True):
        self.config, self.training = config, training
        self.video = CocoVideo(
            config.train_annotations if training else config.val_annotations,
            config.train_images if training else config.val_images,
            config,
        )
        self.windows = []
        span = (config.clip_length - 1) * config.frame_stride + 1
        for seq, frames in sorted(self.video.sequences.items()):
            starts = list(range(0, len(frames) - span + 1, config.clip_step))
            if len(frames) >= span and (not starts or starts[-1] != len(frames) - span):
                starts.append(len(frames) - span)
            for start in starts:
                selected = frames[start : start + span : config.frame_stride]
                gaps = np.diff([f["frame_id"] for f in selected])
                if (gaps <= 0).any() or (gaps > config.max_frame_gap).any():
                    continue
                self.windows.append((seq, selected))
        if not self.windows:
            raise ValueError("No valid contiguous clips; check clip_length/frame metadata/gaps")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        config = self.config
        seed = int.from_bytes(
            hashlib.sha256(f"{config.seed}/{epoch}/{index}".encode()).digest()[:4], "little"
        )
        rng = random.Random(seed)
        cv2.setRNGSeed(seed % (2**31 - 1))
        seq, frames = self.windows[index]
        flip = self.training and rng.random() < config.horizontal_flip
        items = []
        tensors = []
        images = []
        for frame in frames:
            image, annotations = self.video.read(frame)
            rows = self.video.boxes(frame, annotations)
            # Ignore annotations must not become unlabeled YOLO foreground negatives.
            # This follows Falcon's blackout convention and is restricted to training.
            if self.training and self.video.ignored[frame["id"]]:
                original = image.copy()
                for ignored in self.video.ignored[frame["id"]]:
                    x, y, w, h = ignored["bbox"]
                    x0, y0 = max(0, int(x)), max(0, int(y))
                    x1, y1 = min(image.shape[1], int(x + w)), min(image.shape[0], int(y + h))
                    if x1 > x0 and y1 > y0:
                        image[y0:y1, x0:x1] = 0
                for annotation in annotations:
                    x, y, w, h = annotation["bbox"]
                    x0, y0 = max(0, int(x)), max(0, int(y))
                    x1, y1 = min(image.shape[1], int(x + w)), min(image.shape[0], int(y + h))
                    if x1 > x0 and y1 > y0:
                        image[y0:y1, x0:x1] = original[y0:y1, x0:x1]
            if flip:
                image = np.ascontiguousarray(image[:, ::-1])
                rows = [(i, c, np.array([1 - b[0], *b[1:]], np.float32)) for i, c, b in rows]
            tensor, resized = prepare_image(image, config)
            tensors.append(tensor)
            images.append(resized)
            items.append(rows)
        identities = sorted({i for rows in items for i, _, _ in rows})
        # Empty clips remain valid detector-negative examples.
        count = max(1, len(identities))
        lookup = {identity: i for i, identity in enumerate(identities)}
        boxes = torch.zeros(config.clip_length, count, 4)
        present = torch.zeros(config.clip_length, count, dtype=torch.bool)
        labels = torch.zeros(count, dtype=torch.long)
        ids = torch.full((count,), -1, dtype=torch.long)
        for t, rows in enumerate(items):
            for identity, cls, box in rows:
                j = lookup[identity]
                boxes[t, j] = torch.from_numpy(box)
                present[t, j] = True
                labels[j] = cls
                ids[j] = identity
        transforms = np.tile(np.eye(3, dtype=np.float32)[:2], (config.clip_length, 1, 1))
        camera_valid = np.zeros(config.clip_length, bool)
        for t in range(1, len(frames)):
            previous_boxes = boxes[t - 1, present[t - 1]].numpy()
            transforms[t], camera_valid[t] = estimate_affine(images[t - 1], images[t], previous_boxes)
        delta = torch.ones(config.clip_length)
        delta[1:] = torch.tensor(np.diff([f["frame_id"] for f in frames]), dtype=torch.float32)
        alive = torch.zeros_like(present)
        life_known = torch.zeros_like(present)
        sequence_end = self.video.sequences[seq][-1]["frame_id"]
        for identity, j in lookup.items():
            first, last = self.video.lifespans[identity]
            for t, frame in enumerate(frames):
                fid = frame["frame_id"]
                alive[t, j] = first <= fid <= last
                # An identity missing inside its annotated lifespan remains alive.
                # A sequence boundary is censored, not a negative existence label.
                life_known[t, j] = alive[t, j] or (
                    fid >= last + self.config.al_exit_grace
                    and sequence_end >= last + self.config.al_exit_grace
                )
        return dict(
            images=torch.stack(tensors),
            boxes=boxes,
            present=present,
            alive=alive,
            life_known=life_known,
            classes=labels,
            ids=ids,
            affine=torch.from_numpy(transforms),
            camera_valid=torch.from_numpy(camera_valid),
            delta=delta,
            sequence=seq,
            frame_ids=[f["frame_id"] for f in frames],
        )


def collate_clips(items: list[dict]) -> dict:
    count = max(x["boxes"].shape[1] for x in items)
    result = {
        key: torch.stack([x[key] for x in items]) for key in ("images", "affine", "camera_valid", "delta")
    }
    for key in ("boxes", "present", "alive", "life_known", "classes", "ids"):
        arrays = []
        for item in items:
            tensor = item[key]
            shape = list(tensor.shape)
            axis = 1 if key in ("boxes", "present", "alive", "life_known") else 0
            shape[axis] = count
            padded = tensor.new_full(shape, -1 if key == "ids" else 0)
            index = [slice(None)] * tensor.ndim
            index[axis] = slice(0, tensor.shape[axis])
            padded[tuple(index)] = tensor
            arrays.append(padded)
        result[key] = torch.stack(arrays)
    result["sequence"] = [x["sequence"] for x in items]
    result["frame_ids"] = [x["frame_ids"] for x in items]
    return result


class ClipSampler(Sampler):
    """Epoch is part of each sample key, including inside persistent workers."""

    def __init__(self, dataset: VideoClips, epoch: int, start: int = 0):
        order = list(range(len(dataset)))
        random.Random(dataset.config.seed + epoch).shuffle(order)
        self.keys = [(epoch, i) for i in order[start:]]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)

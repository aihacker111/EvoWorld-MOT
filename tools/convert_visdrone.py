#!/usr/bin/env python3
"""Convert raw VisDrone-MOT using image headers, with flushed progress logs."""

import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import time
import cv2
from PIL import Image

NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def image_size(path: Path, verify_images: bool = False):
    if verify_images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        return image.shape[:2]
    # Lazy header parsing: no pixel decoding, and no assumption of constant sequence size.
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274, 1) in (5, 6, 7, 8):
            width, height = height, width
    return height, width


def resolve_root(root: Path, split: str | None = None, max_depth: int = 6) -> Path:
    """Resolve archive wrappers using a bounded directory search, skipping image trees."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if split not in (None, "train", "val"):
        raise ValueError("split must be train or val")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    pending = deque([(root, 0)])
    visited, candidates = set(), []
    while pending:
        current, depth = pending.popleft()
        canonical = current.resolve()
        if canonical in visited:
            continue
        visited.add(canonical)
        if (current / "sequences").is_dir() and (current / "annotations").is_dir():
            candidates.append(current)
            continue  # Never enumerate files under sequences/ or annotations/.
        if depth < max_depth:
            pending.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir())
                if child.is_dir() and not child.name.startswith(".")
            )
    if split is not None:

        def matches(path):
            parts = (root.name, *path.relative_to(root).parts)
            return any(
                part.lower() == split or part.lower().endswith((f"-{split}", f"_{split}")) for part in parts
            )

        labeled = [candidate for candidate in candidates if matches(candidate)]
        # A directly supplied root is already explicit, even with a generic name.
        named_split = any(
            root.name.lower() == name or root.name.lower().endswith((f"-{name}", f"_{name}"))
            for name in ("train", "val")
        )
        if not labeled and candidates == [root] and not named_split:
            labeled = candidates
        candidates = labeled
    if not candidates:
        raise FileNotFoundError(
            f"No annotations/ + sequences/ root for split={split!r} "
            f"within {max_depth} levels of {root}; point --root at the inner dataset folder"
        )
    if len(candidates) > 1:
        choices = "\n".join(str(path) for path in candidates)
        raise ValueError(
            f"Multiple dataset roots found; use --split train/val or an exact --root:\n{choices}"
        )
    return candidates[0]


def convert(
    root: Path, output: Path, progress_every: int = 500, verify_images: bool = False, split: str | None = None
):
    root, output = resolve_root(root, split), Path(output)
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    sequences = sorted(p for p in (root / "sequences").iterdir() if p.is_dir())
    started = time.monotonic()
    print(
        f"Converting {root}: {len(sequences)} sequences; image mode={'decode' if verify_images else 'header'}",
        flush=True,
    )
    payload = dict(
        images=[],
        annotations=[],
        categories=[{"id": i + 1, "name": n} for i, n in enumerate(NAMES)],
    )
    for si, sequence in enumerate(sequences, 1):
        print(f"[{si}/{len(sequences)}] Reading {sequence.name}", flush=True)
        annotation_path = root / "annotations" / f"{sequence.name}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        annotations = defaultdict(list)
        with annotation_path.open() as file:
            for row in csv.reader(file):
                if not row:
                    continue
                if len(row) != 10:
                    raise ValueError(f"{annotation_path}: expected 10 columns, got {len(row)}")
                values = [float(x) for x in row]
                annotations[int(values[0])].append(values)
        paths = sorted(p for p in sequence.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        image_frames = set()
        for fi, path in enumerate(paths, 1):
            frame = int(path.stem)
            if frame in image_frames:
                raise ValueError(f"Duplicate frame stem in {sequence}: {frame}")
            image_frames.add(frame)
            h, w = image_size(path, verify_images)
            image_id = len(payload["images"]) + 1
            payload["images"].append(
                dict(
                    id=image_id,
                    file_name=f"{sequence.name}/{path.name}",
                    seq_id=sequence.name,
                    frame_id=frame,
                    height=h,
                    width=w,
                )
            )
            for _, tid, x, y, bw, bh, score, cat, trunc, occlusion in annotations[frame]:
                if bw <= 0 or bh <= 0:
                    continue
                ignore = score == 0 or int(cat) not in range(1, 11)
                if not ignore and (tid < 0 or int(tid) != tid):
                    raise ValueError(f"Invalid track ID in {annotation_path}: {tid}")
                payload["annotations"].append(
                    dict(
                        id=len(payload["annotations"]) + 1,
                        image_id=image_id,
                        track_id=int(tid),
                        category_id=int(cat),
                        bbox=[x, y, bw, bh],
                        area=bw * bh,
                        iscrowd=0,
                        ignore=int(ignore),
                        truncation=trunc,
                        occlusion=occlusion,
                    )
                )
            if fi % progress_every == 0 or fi == len(paths):
                print(
                    f"[{si}/{len(sequences)}] {sequence.name}: {fi}/{len(paths)} frames | "
                    f"total={len(payload['images'])} | elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if set(annotations) - image_frames:
            raise ValueError(f"Annotations reference missing frames in {sequence}")
    if not payload["images"]:
        raise ValueError("No sequences/images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output} ...", flush=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(output)
    print(
        f"{output}: {len(payload['images'])} frames, {len(payload['annotations'])} annotations",
        flush=True,
    )
    print(f"Use image root: {root / 'sequences'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("train", "val"), help="Select a split inside a wrapped dataset folder"
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every image to check pixel data; slower",
    )
    arguments = parser.parse_args()
    convert(arguments.root, arguments.out, arguments.progress_every, arguments.verify_images, arguments.split)

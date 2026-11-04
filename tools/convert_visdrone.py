#!/usr/bin/env python3
"""Convert one RAW VisDrone-MOT split; reference original frames without copying them."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2

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


def convert(root: Path, output: Path):
    payload = dict(
        images=[], annotations=[], categories=[{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]
    )
    for sequence in sorted((root / "sequences").iterdir()):
        if not sequence.is_dir():
            continue
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
        image_frames = set()
        for path in sorted(sequence.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            frame = int(path.stem)
            if frame in image_frames:
                raise ValueError(f"Duplicate frame stem in {sequence}: {frame}")
            image_frames.add(frame)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]
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
        if set(annotations) - image_frames:
            raise ValueError(f"Annotations reference missing frames in {sequence}")
    if not payload["images"]:
        raise ValueError("No sequences/images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload))
    print(f"{output}: {len(payload['images'])} frames, {len(payload['annotations'])} annotations")
    print(f"Use image root: {root / 'sequences'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    convert(arguments.root, arguments.out)

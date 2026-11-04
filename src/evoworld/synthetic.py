"""Small deterministic moving/crossing-object data for wiring tests, not a benchmark."""

import json
from pathlib import Path
import cv2
import numpy as np


def make_dataset(out, frames=18, width=96, height=64):
    out = Path(out)
    image_root = out / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 11)],
    }
    rng = np.random.default_rng(123)
    background = rng.integers(0, 35, (height, width, 3), dtype=np.uint8)
    for t in range(frames):
        image = background.copy()
        fid = t + 1
        payload["images"].append(
            dict(
                id=fid,
                file_name=f"{fid:06d}.jpg",
                height=height,
                width=width,
                seq_id="crossing",
                frame_id=fid,
            )
        )
        for identity in range(2):
            x = (
                int((0.15 + 0.5 * t / max(1, frames - 1)) * width)
                if identity == 0
                else int((0.75 - 0.5 * t / max(1, frames - 1)) * width)
            )
            y, w, h = int((0.25 + 0.15 * identity) * height), max(5, width // 9), max(7, height // 4)
            color = (30, 220, 230) if identity == 0 else (210, 100, 40)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
            payload["annotations"].append(
                dict(
                    id=len(payload["annotations"]) + 1,
                    image_id=fid,
                    category_id=4,
                    track_id=identity,
                    bbox=[x, y, w, h],
                    area=w * h,
                    iscrowd=0,
                )
            )
        if not cv2.imwrite(str(image_root / f"{fid:06d}.jpg"), image):
            raise OSError("Failed writing synthetic frame")
    path = out / "annotations.json"
    path.write_text(json.dumps(payload))
    return path, image_root

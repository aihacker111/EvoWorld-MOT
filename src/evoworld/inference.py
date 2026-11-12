"""Streaming video/image-directory tracking with a rendered preview and MOT export."""

import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from evoworld.camera import estimate_affine
from evoworld.config import atomic_json
from evoworld.data import prepare_image
from evoworld.tracking import EcologyTracker


def open_preview(out, fps, width, height):
    """Some macOS OpenCV builds lack an MP4 encoder but support MJPEG/AVI."""
    for codec, suffix in (("mp4v", "mp4"), ("MJPG", "avi")):
        path = Path(out) / f"preview.{suffix}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, path
        writer.release()
    raise OSError("No MP4/MJPEG encoder available in OpenCV; install an OpenCV build with video encoding")


def identity_color(identity: int) -> tuple[int, int, int]:
    return ((identity * 67) % 200 + 55, (identity * 113) % 200 + 55, (identity * 37) % 200 + 55)


def draw_track(image, identity, x, y, w, h, category, confidence, thickness=2):
    """Draw one track box (pixel x, y, w, h) with its ID, category id and score."""
    color = identity_color(identity)
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    cv2.putText(
        image,
        f"{identity} c{category} {confidence:.2f}",
        (int(x), max(12, int(y) - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )


def frames(source):
    source = Path(source)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not paths:
            raise ValueError("No images found")
        for i, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            yield i, image
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")
        try:
            i = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                i += 1
                yield i, image
        finally:
            capture.release()


@torch.no_grad()
def track(model, source, out, device="cpu", fps=25.0):
    if model.config.backend != "falcon":
        raise ValueError("Real tracking requires the Falcon detector")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tracker = EcologyTracker(model)
    previous, previous_boxes = None, np.empty((0, 4), np.float32)
    writer = None
    preview_path = None
    started, count = time.monotonic(), 0
    try:
        with (
            (out / "tracks.txt").open("w") as result,
            (out / "classes.jsonl").open("w") as classes_file,
            (out / "life.jsonl").open("w") as life_file,
        ):
            for frame_id, image in frames(source):
                tensor, resized = prepare_image(image, model.config)
                features = model.perception(tensor[None].to(device))
                detections = model.perception.detections(features)[0]
                affine = None
                if previous is not None:
                    transform, valid = estimate_affine(previous, resized, previous_boxes)
                    if valid:
                        affine = torch.from_numpy(transform).to(device)
                rows = tracker.step(detections, features.appearance, features.camera, frame_id, affine)
                if hasattr(tracker, "last_diagnostics"):
                    life_file.write(json.dumps(dict(frame=frame_id, **tracker.last_diagnostics)) + "\n")
                height, width = image.shape[:2]
                if writer is None:
                    writer, preview_path = open_preview(out, fps, width, height)
                    video_shape = image.shape[:2]
                if image.shape[:2] != video_shape:
                    raise ValueError("All frames in a stream must have the same dimensions")
                for identity, _, box, confidence, cls in rows:
                    x, y, w, h = box.cpu().tolist()
                    x, y, w, h = (x - w / 2) * width, (y - h / 2) * height, w * width, h * height
                    result.write(
                        f"{frame_id},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{confidence:.5f},-1,-1,-1\n"
                    )
                    category = model.config.category_ids[cls]
                    classes_file.write(
                        json.dumps(dict(frame=frame_id, id=identity, category_id=category)) + "\n"
                    )
                    draw_track(image, identity, x, y, w, h, category, confidence)
                # Rows of tentative frames released at confirmation (MOT readers group by frame).
                for past_frame, identity, box, confidence, cls in getattr(tracker, "last_backfill", []):
                    x, y, w, h = box.cpu().tolist()
                    x, y, w, h = (x - w / 2) * width, (y - h / 2) * height, w * width, h * height
                    result.write(
                        f"{past_frame},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{confidence:.5f},-1,-1,-1\n"
                    )
                    category = model.config.category_ids[cls]
                    classes_file.write(
                        json.dumps(dict(frame=past_frame, id=identity, category_id=category)) + "\n"
                    )
                writer.write(image)
                previous, previous_boxes = resized, detections["boxes"].cpu().numpy()
                count += 1
    finally:
        if writer is not None:
            writer.release()
    seconds = time.monotonic() - started
    summary = dict(
        frames=count,
        seconds=seconds,
        fps=count / max(seconds, 1e-8),
        preview=str(preview_path) if preview_path else None,
        results=str(out / "tracks.txt"),
    )
    atomic_json(out / "summary.json", summary)
    return summary

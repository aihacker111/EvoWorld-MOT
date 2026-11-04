#!/usr/bin/env python3
"""Compact training console; raw training JSONL is left untouched by this reader."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys


COLUMNS = [
    ("Epoch", 5),
    ("Update", 7),
    ("Clips", 11),
    ("Loss", 7),
    ("Det", 6),
    ("ReID", 6),
    ("Assoc", 6),
    ("World", 7),
    ("AL", 6),
    ("Grad", 6),
    ("clip/s", 6),
    ("ETA ep", 7),
]


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def number(value, precision=3):
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(value):
        return str(value)
    text = f"{value:.{precision}f}"
    return text if len(text) <= 7 else f"{value:.1e}"


class TrainingConsole:
    def __init__(self, stream=None, every=10, mode="pretty"):
        if every < 1:
            raise ValueError("log-every must be positive")
        self.stream = sys.stdout if stream is None else stream
        self.every, self.mode = every, mode
        self.run = None
        self.config = {}
        self.samples = 0
        self.epochs = None
        self.last = None
        self.last_printed = None
        self.last_printed_seconds = None
        self.rate = None
        self.records_seen = 0
        self.table_epoch = None
        self.table_rows = 0
        self.checkpoint = None

    def write(self, text):
        self.stream.write(text + "\n")
        self.stream.flush()

    def consume(self, line):
        if self.mode == "json":
            self.stream.write(line)
            self.stream.flush()
            return
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            if line.strip().startswith("Checkpoint:"):
                self.checkpoint = line.strip().partition(":")[2].strip()
            else:
                self.stream.write(line)
                self.stream.flush()
            return
        if not isinstance(record, dict):
            if self.mode != "quiet":
                self.write(line.rstrip("\n"))
            return
        if "run" in record and "parameters" in record:
            self.start(record)
        elif record.get("event", "").startswith("validation_"):
            if self.mode != "quiet":
                self.validation(record)
        elif record.get("event") == "amp_overflow":
            if self.mode != "quiet":
                self.write(
                    f"AMP skip | epoch {record.get('epoch', '?')} | update {record.get('updates', '?')} "
                    f"| scale {record.get('scale', 0):,.0f}"
                )
                self.table_epoch = None
        elif {"loss", "updates", "epoch"} <= record.keys():
            self.update(record)
        else:
            self.write(line.rstrip("\n"))

    def validation(self, record):
        event = record["event"]
        if event == "validation_start":
            if self.last is not None and self.last_printed != (self.last["epoch"], self.last["updates"]):
                self.row(self.last)
            limit = record.get("max_frames")
            scope = f"up to {limit:,} frames" if limit is not None else "full dataset"
            kind = "SMOKE" if record.get("smoke_test") else "FULL"
            self.write(f"Validation {kind} | update {record['updates']} | {scope}")
        elif event == "validation_progress":
            self.write(f"VAL | {record['frames']:,}/{record['total_frames']:,} frames | {record['sequence']}")
        elif event == "validation_end":
            scope = "partial" if record.get("truncated") else "full"
            self.write(
                f"Validation OK | IDF1 {100 * record['idf1']:.2f}% | MOTA {100 * record['mota']:.2f}% "
                f"| IDSW {record['id_switches']} | {record['frames']:,} frames ({scope}) "
                f"| {duration(record['seconds'])}"
            )
            self.write(f"Validation results: {record['out']}")
            self.rate = None
            self.last = None
        self.table_epoch = None

    def start(self, record):
        self.run = record["run"]
        try:
            metadata = json.loads((Path(self.run) / "run.json").read_text())
        except (OSError, ValueError):
            metadata = {}
        self.config = metadata.get("config", {})
        self.epochs = self.config.get("epochs")
        world = record.get("world_size", metadata.get("distributed", {}).get("world_size", 1))
        self.samples = record.get("samples_per_epoch", record.get("clips", 0))
        devices = record.get("devices", [record.get("device", "?")])
        effective = record.get("effective_batch")
        if effective is None and self.config:
            effective = self.config.get("batch_size", 1) * self.config.get("grad_accumulation", 1) * world
        params = sum(record["parameters"].values())
        if self.mode == "quiet":
            return
        self.write("EvoWorld-MOT | TRAIN")
        hardware = f"{world} GPU" if any(str(d).startswith("cuda") for d in devices) else f"{world} process"
        self.write(
            f"{hardware} [{', '.join(map(str, devices))}] | {params / 1e6:.2f}M params | "
            f"batch {effective if effective is not None else '?'} | {self.samples:,} clips/epoch"
        )
        self.write(f"Run: {self.run}")
        if self.config.get("tracker_mode", "al") == "al":
            self.write(
                f"AL warmup: policy {self.config.get('al_min_examples', 256):,} | "
                f"critic {self.config.get('al_critic_min_examples', 64):,} examples"
            )

    def al_status(self, record):
        if self.config.get("tracker_mode") == "legacy":
            return "off"
        policy, critic = record.get("al_policy_total"), record.get("al_critic_total")
        if policy is None or critic is None:
            return "--"
        ready = policy >= self.config.get("al_min_examples", 256) and critic >= self.config.get(
            "al_critic_min_examples", 64
        )
        return "ready" if ready else "warm"

    def update(self, record):
        self.records_seen += 1
        epoch, seen, seconds = record["epoch"], record.get("clips_seen", 0), record.get("seconds", 0.0)
        if self.last and self.last["epoch"] == epoch:
            delta_time = seconds - self.last.get("seconds", 0.0)
            delta_clips = seen - self.last.get("clips_seen", 0)
            if delta_time > 0 and delta_clips > 0:
                instantaneous = delta_clips / delta_time
                self.rate = instantaneous if self.rate is None else 0.2 * instantaneous + 0.8 * self.rate
            elif delta_time < 0:
                self.rate = None
        else:
            self.rate = None
        self.last = record
        at_end = self.samples > 0 and seen >= self.samples
        heartbeat = self.last_printed_seconds is not None and seconds - self.last_printed_seconds >= 30
        show = (
            self.records_seen <= 3
            or record["updates"] % self.every == 0
            or self.table_epoch != epoch
            or at_end
            or heartbeat
        )
        if self.mode == "pretty" and show:
            self.row(record)

    def row(self, record):
        epoch = record["epoch"]
        if self.table_epoch != epoch or self.table_rows >= 25:
            self.write(" ".join(name.rjust(width) for name, width in COLUMNS))
            self.write(" ".join("-" * width for _, width in COLUMNS))
            self.table_epoch, self.table_rows = epoch, 0
        seen = record.get("clips_seen", 0)
        remaining = max(0, self.samples - seen) / self.rate if self.samples and self.rate else None
        epoch_text = f"{epoch}/{self.epochs}" if self.epochs else str(epoch)
        clips_text = f"{seen}/{self.samples}" if self.samples else str(seen)
        cells = [
            epoch_text,
            str(record["updates"]),
            clips_text,
            number(record.get("loss")),
            number(record.get("detector")),
            number(record.get("reid")),
            number(record.get("association")),
            number(record.get("world")),
            self.al_status(record),
            number(record.get("grad_norm"), 2),
            number(self.rate, 2),
            duration(remaining),
        ]
        self.write(" ".join(value.rjust(width) for value, (_, width) in zip(cells, COLUMNS)))
        self.table_rows += 1
        self.last_printed = (epoch, record["updates"])
        self.last_printed_seconds = record.get("seconds", 0.0)

    def finish(self, returncode):
        if self.mode != "pretty" or self.run is None:
            return
        if self.last and self.last_printed != (self.last["epoch"], self.last["updates"]):
            self.row(self.last)
        if returncode is None:
            self.write("Log stream ended")
        elif returncode:
            self.write(f"Training stopped | exit code {returncode}")
        else:
            self.write("Run finished")
        checkpoint = self.checkpoint or str(Path(self.run) / "last.pt")
        if self.checkpoint is not None or Path(checkpoint).is_file():
            self.write(f"Checkpoint: {checkpoint}")
        self.write(f"Metrics: {Path(self.run) / 'metrics.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--format", choices=("pretty", "json", "quiet"), default="pretty")
    args = parser.parse_args()
    console = TrainingConsole(every=args.every, mode=args.format)
    for line in sys.stdin:
        console.consume(line)
    console.finish(None)


if __name__ == "__main__":
    main()

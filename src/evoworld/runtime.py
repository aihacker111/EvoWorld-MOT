"""Reproducible runtime and atomic, provenance-checked checkpointing."""

from __future__ import annotations
from pathlib import Path
import platform
import random
import numpy as np
import torch
from evoworld.config import Config, sha256


def setup(config):
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config.device != "auto":
        return torch.device(config.device)
    # CPU is the reproducible Mac default; CUDA AMP is used on training servers.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
    }


def rng_state(device=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            {"device": device.index, "state": torch.cuda.get_rng_state(device)}
            if device is not None and device.type == "cuda"
            else torch.cuda.get_rng_state_all()
            if device is None and torch.cuda.is_available()
            else None
        ),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        if isinstance(state["cuda"], dict):
            torch.cuda.set_rng_state(state["cuda"]["state"], state["cuda"]["device"])
        else:
            torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scaler, progress, provenance, distributed=None):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 2,
            "config": model.config.dictionary(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "progress": progress,
            "provenance": provenance,
            "environment": environment(),
            "rng": distributed["rng_by_rank"][0] if distributed else rng_state(),
            "distributed": distributed,
        },
        temporary,
    )
    temporary.replace(path)


def load_for_inference(path, device="cpu"):
    from evoworld.models.system import EvoWorld

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError(
            "Expected an EvoWorld checkpoint; use the Falcon detector only to initialize training"
        )
    configuration = dict(checkpoint["config"])
    if checkpoint["format_version"] == 1:
        configuration["tracker_mode"] = "legacy"
    config = Config(**configuration).validate()
    model = EvoWorld(config, initialize=False)
    checkpoint["compatibility"] = transfer_state(model, checkpoint["model"])
    return model.to(device).eval(), checkpoint


def check_resume(checkpoint, config, provenance):
    if checkpoint.get("format_version") != 2:
        raise ValueError(
            "v1 checkpoints cannot resume AL optimization; use --init-checkpoint for a new AL run"
        )
    # Changing these does not change the optimization or sample/augmentation stream.
    mutable = {"out", "epochs", "workers", "device", "validate_every"}
    old = checkpoint["config"]
    old = Config(**old).dictionary()
    changed = [k for k, v in config.dictionary().items() if k not in mutable and old[k] != v]
    # Epoch count changes the teacher-forcing schedule, hence must remain fixed in practice.
    if old["epochs"] != config.epochs:
        changed.append("epochs (teacher-forcing schedule)")
    if changed or checkpoint["provenance"] != provenance:
        raise ValueError(
            f"Resume provenance/config mismatch: {changed}; use a new run for changed experiments"
        )


# Parameters introduced after v0.2 that older checkpoints legitimately lack.
NEW_PARAMETERS = ("association.score_weights",)


def transfer_state(model, state):
    """Load weights from an older architecture without changing its function.

    * identical tensors are copied;
    * a Linear weight whose input grew by appended features keeps old columns and
      gets zero new columns (the new inputs start with no effect);
    * AL weights (v1 checkpoints) and listed new parameters keep their initialization.
    Anything else is an architecture mismatch.
    """
    expected = model.state_dict()
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    allowed = [k for k in missing if k.startswith("al.") or k in NEW_PARAMETERS]
    expanded, wrong = [], []
    merged = dict(expected)
    for key in set(state) & set(expected):
        old, new = state[key], expected[key]
        if old.shape == new.shape:
            merged[key] = old
        elif old.ndim == 2 and new.ndim == 2 and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
            grown = new.new_zeros(new.shape)
            grown[:, : old.shape[1]] = old
            merged[key] = grown
            expanded.append(key)
        else:
            wrong.append(key)
    if unexpected or wrong or len(allowed) != len(missing):
        raise ValueError(
            "Warm-start architecture mismatch: "
            f"unexpected={unexpected[:3]}, missing={[k for k in missing if k not in allowed][:3]}, "
            f"shape={wrong[:3]}"
        )
    model.load_state_dict(merged, strict=True)
    return dict(newly_initialized=missing, expanded_inputs=sorted(expanded))


def initialize_from_checkpoint(model, path):
    """Explicit weight transfer; never pretend old optimizer/RNG states fit a new algorithm."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("Expected an EvoWorld v1/v2 checkpoint for --init-checkpoint")
    report = transfer_state(model, checkpoint["model"])
    return dict(
        path=str(path),
        sha256=sha256(path),
        format_version=checkpoint["format_version"],
        **report,
    )

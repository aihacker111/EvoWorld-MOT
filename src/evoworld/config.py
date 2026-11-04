"""Explicit, validated configuration; no silent unknown options."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    detector_yaml: str = "configs/falcon_detector.yaml"
    detector_checkpoint: str = "weights/falcon_detector.pth"
    detector_scale: str = "s"
    num_classes: int = 10
    category_ids: list[int] = field(default_factory=lambda: list(range(1, 11)))
    backend: str = "falcon"
    tracker_mode: str = "al"
    preprocess: str = "imagenet_rgb"
    width: int = 960
    height: int = 544
    embedding_dim: int = 128
    hidden_dim: int = 128
    stochastic_dim: int = 16
    camera_dim: int = 64
    neighbors: int = 8
    motion_modes: int = 3
    train_annotations: str = ""
    train_images: str = ""
    val_annotations: str = ""
    val_images: str = ""
    clip_length: int = 8
    burn_in: int = 2
    horizon: int = 4
    clip_step: int = 4
    frame_stride: int = 1
    max_frame_gap: int = 4
    synthetic_dropout: float = 0.15
    box_noise: float = 0.01
    proposal_probability: float = 0.5
    proposal_iou: float = 0.5
    horizontal_flip: float = 0.5
    batch_size: int = 1
    workers: int = 2
    epochs: int = 40
    learning_rate: float = 0.0001
    perception_lr: float = 0.000005
    weight_decay: float = 0.0001
    grad_accumulation: int = 4
    grad_clip: float = 5.0
    amp: bool = True
    seed: int = 2026
    threads: int = 2
    device: str = "auto"
    out: str = "runs/full"
    burn_in_no_grad: bool = True
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    warmup_epochs: int = 3
    reid_weight: float = 0.5
    association_weight: float = 1.0
    world_weight: float = 1.0
    camera_weight: float = 0.2
    effect_weight: float = 0.5
    stability_weight: float = 0.1
    kl_weight: float = 0.01
    free_nats: float = 1.0
    probe_every: int = 4
    probe_horizon: int = 4
    effect_min_examples: int = 64
    stability_limit: float = 0.8
    effect_margin: float = 0.25
    ecology_budget: int = 8
    mutation_edges: int = 2
    conf_threshold: float = 0.35
    low_threshold: float = 0.1
    birth_threshold: float = 0.5
    nms_iou: float = 0.6
    max_detections: int = 300
    max_age: int = 30
    match_threshold: float = -1.5
    uncertain_margin: float = 0.8
    effect_score_weight: float = 1.0
    likelihood_temperature: float = 1.0
    validate_every: int = 1
    checkpoint_every: int = 100
    al_weight: float = 0.5
    al_compute_budget: int = 32
    al_credit_capacity: float = 6.0
    al_credit_refill: float = 1.0
    al_planning_agents: int = 6
    al_plan_horizon: int = 3
    al_min_examples: int = 256
    al_critic_min_examples: int = 64
    al_confirm_hits: int = 2
    al_tentative_age: int = 3
    al_max_dormant_age: int = 90
    al_exit_grace: int = 3
    al_terminate_probability: float = 0.7
    al_existence_threshold: float = 0.25
    al_birth_probability: float = 0.35
    al_match_probability: float = 0.1
    al_acquisition_cost: float = 0.05
    al_risk_weight: float = 0.5
    al_forecast_weight: float = 0.2
    al_burst_probability: float = 0.3
    al_burst_length: int = 3
    al_memory_corruption: float = 0.15

    def validate(self) -> "Config":
        if self.tracker_mode not in ("al", "legacy"):
            raise ValueError("tracker_mode must be al or legacy")
        if self.backend not in ("falcon", "tiny"):
            raise ValueError("backend must be falcon or tiny (synthetic tests only)")
        if self.preprocess not in ("imagenet_rgb", "rgb01"):
            raise ValueError("preprocess must be imagenet_rgb or rgb01")
        if self.width % 32 or self.height % 32:
            raise ValueError("Image dimensions must be divisible by 32")
        if len(self.category_ids) != self.num_classes or len(set(self.category_ids)) != self.num_classes:
            raise ValueError("category_ids must map each model class to one distinct dataset category")
        for key in (
            "width",
            "height",
            "embedding_dim",
            "hidden_dim",
            "stochastic_dim",
            "camera_dim",
            "neighbors",
            "motion_modes",
            "epochs",
            "batch_size",
            "grad_accumulation",
            "horizon",
            "probe_horizon",
            "probe_every",
            "ecology_budget",
            "clip_step",
            "frame_stride",
            "max_frame_gap",
            "max_detections",
            "threads",
            "validate_every",
            "checkpoint_every",
            "al_plan_horizon",
            "al_min_examples",
            "al_critic_min_examples",
            "al_confirm_hits",
            "al_tentative_age",
            "al_max_dormant_age",
            "al_exit_grace",
            "al_burst_length",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.burn_in < 0 or self.clip_length < self.burn_in + 2:
            raise ValueError("clip_length must cover burn-in and at least two learning frames")
        if self.horizon >= self.clip_length - self.burn_in:
            raise ValueError("horizon must be shorter than the learning segment")
        for key in (
            "synthetic_dropout",
            "horizontal_flip",
            "teacher_forcing_start",
            "teacher_forcing_end",
            "proposal_probability",
            "proposal_iou",
            "al_terminate_probability",
            "al_existence_threshold",
            "al_burst_probability",
            "al_memory_corruption",
            "al_birth_probability",
            "al_match_probability",
        ):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if not 0 <= self.low_threshold <= self.conf_threshold <= self.birth_threshold <= 1:
            raise ValueError("Require low_threshold <= conf_threshold <= birth_threshold <= 1")
        if self.workers < 0 or self.box_noise < 0 or self.effect_min_examples < 1:
            raise ValueError("Invalid workers, box noise, or effect readiness threshold")
        for key in ("learning_rate", "perception_lr", "grad_clip", "likelihood_temperature"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "weight_decay",
            "reid_weight",
            "association_weight",
            "world_weight",
            "camera_weight",
            "effect_weight",
            "stability_weight",
            "kl_weight",
            "free_nats",
            "effect_margin",
            "effect_score_weight",
            "uncertain_margin",
            "mutation_edges",
            "warmup_epochs",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 < self.nms_iou <= 1 or self.max_age < 0 or self.num_classes < 1:
            raise ValueError("Invalid NMS IoU, max_age or num_classes")
        if self.al_compute_budget < 0 or self.al_planning_agents < 0:
            raise ValueError("AL compute budget and planning agents must be nonnegative")
        if self.al_credit_capacity <= 0 or self.al_credit_refill < 0:
            raise ValueError("Invalid per-agent compute credits")
        if min(self.al_weight, self.al_acquisition_cost, self.al_risk_weight, self.al_forecast_weight) < 0:
            raise ValueError("AL weights and costs must be nonnegative")
        return self

    def dictionary(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**raw).validate()


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config: Config) -> dict:
    sources = sorted((ROOT / "src" / "evoworld").rglob("*.py"))
    return {
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "detector_yaml": sha256(resolve(config.detector_yaml)),
        "initial_weights": sha256(resolve(config.detector_checkpoint))
        if config.backend == "falcon"
        else None,
        "annotations": {
            key: sha256(resolve(getattr(config, key)))
            for key in ("train_annotations", "val_annotations")
            if getattr(config, key)
        },
    }


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

"""Deterministic optimizer ownership and explicit module learning rates."""

import math
import torch


MODULES = ("detector", "reid", "world", "camera", "association", "al")


def module_name(parameter_name):
    if parameter_name.startswith("perception.reid."):
        return "reid"
    if parameter_name.startswith("perception."):
        return "detector"
    name = parameter_name.partition(".")[0]
    if name not in MODULES:
        raise ValueError(f"Optimizer has no owner for parameter {parameter_name}")
    return name


def module_rate(config, name):
    if name == "detector":
        return config.perception_lr
    override = getattr(config, f"{name}_lr")
    return config.learning_rate if override is None else override


def build_optimizer(model, config):
    """Keep the original two-group layout unless module overrides are requested."""
    overrides = any(getattr(config, f"{name}_lr") is not None for name in MODULES[1:])
    if not overrides:
        perception = list(model.perception.parameters())
        perception_ids = {id(p) for p in perception}
        reid_ids = {id(p) for p in model.perception.reid.parameters()}
        groups = [
            dict(
                name="detector",
                params=[p for p in perception if id(p) not in reid_ids],
                lr=config.perception_lr,
                base_lr=config.perception_lr,
            ),
            dict(
                name="new",
                params=[p for p in model.parameters() if id(p) not in perception_ids or id(p) in reid_ids],
                lr=config.learning_rate,
                base_lr=config.learning_rate,
            ),
        ]
    else:
        owners = {name: [] for name in MODULES}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                owners[module_name(name)].append(parameter)
        groups = [
            dict(
                name=name, params=parameters, lr=module_rate(config, name), base_lr=module_rate(config, name)
            )
            for name, parameters in owners.items()
            if parameters
        ]
    if config.no_decay_norm_bias:
        groups = split_decay(groups)
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def split_decay(groups):
    """Biases, norm affines and scalar weights get no decay; same LR/schedule as their module."""
    result = []
    for group in groups:
        decay = [p for p in group["params"] if p.ndim > 1]
        plain = [p for p in group["params"] if p.ndim <= 1]
        if decay:
            result.append(dict(group, params=decay))
        if plain:
            result.append(dict(group, params=plain, name=group["name"] + "/no_decay", weight_decay=0.0))
    return result


def learning_rate_factor(config, epoch):
    """Zero-based epoch schedule: original warmup, then constant or cosine decay.

    The final warmup epoch is the cosine peak. Without warmup, epoch zero
    is the peak. The last training epoch reaches min_lr_ratio exactly.
    A run that ends during warmup simply retains the original warmup.
    """
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if epoch < config.warmup_epochs:
        return (epoch + 1) / config.warmup_epochs
    if config.lr_scheduler == "constant":
        return 1.0
    peak = max(0, config.warmup_epochs - 1)
    span = max(1, config.epochs - 1 - peak)
    progress = min(1.0, max(0.0, (epoch - peak) / span))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def set_epoch_learning_rates(optimizer, config, epoch):
    """Absolute schedule from saved epoch; resume has no mutable scheduler counter."""
    factor = learning_rate_factor(config, epoch)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def learning_rates(optimizer):
    """One entry per module; a module's no-decay group always shares its LR."""
    return {
        group["name"]: group["lr"] for group in optimizer.param_groups if not group["name"].endswith("/no_decay")
    }

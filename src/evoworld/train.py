"""Joint training with differential learning rates and exact optimizer-boundary resumes."""

from __future__ import annotations
import contextlib
import json
import time
import torch
from torch.utils.data import DataLoader
from evoworld.config import Config, resolve, fingerprint, atomic_json
from evoworld.data import VideoClips, ClipSampler, collate_clips
from evoworld.models.system import EvoWorld
from evoworld.objective import training_objective
from evoworld.runtime import (
    setup,
    environment,
    restore_rng,
    save_checkpoint,
    check_resume,
    initialize_from_checkpoint,
)


def train(config: Config, resume=None, max_steps=None, init_checkpoint=None):
    if resume and init_checkpoint:
        raise ValueError("Choose --resume or --init-checkpoint")
    device = setup(config)
    out = resolve(config.out)
    if out.exists() and (out / "last.pt").exists() and not resume:
        raise FileExistsError(f"{out} already has a checkpoint; use --resume or a different --out")
    out.mkdir(parents=True, exist_ok=True)
    dataset = VideoClips(config)
    provenance = fingerprint(config)
    model = EvoWorld(config, initialize=not bool(resume)).to(device).train()
    if init_checkpoint:
        provenance["warmstart"] = initialize_from_checkpoint(model, resolve(str(init_checkpoint)))
    perception_parameters = list(model.perception.parameters())
    perception_ids = {id(p) for p in perception_parameters}
    # New ReID weights use the new-module LR while pretrained detector uses the smaller LR.
    reid_ids = {id(p) for p in model.perception.reid.parameters()}
    pretrained = [p for p in perception_parameters if id(p) not in reid_ids]
    new = [p for p in model.parameters() if id(p) not in perception_ids or id(p) in reid_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": pretrained, "lr": config.perception_lr, "base_lr": config.perception_lr},
            {"params": new, "lr": config.learning_rate, "base_lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    progress = {"epoch": 0, "offset": 0, "updates": 0, "micro_steps": 0, "best_idf1": -1.0}
    if resume:
        checkpoint = torch.load(resolve(str(resume)), map_location="cpu", weights_only=False)
        if "warmstart" in checkpoint["provenance"]:
            provenance["warmstart"] = checkpoint["provenance"]["warmstart"]
        check_resume(checkpoint, config, provenance)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        progress.update(checkpoint["progress"])
        restore_rng(checkpoint["rng"])
    metadata = {
        "config": config.dictionary(),
        "provenance": provenance,
        "environment": environment(),
        "parameters": model.parameter_report(),
        "detector_load": model.perception.load_report,
        "device": str(device),
    }
    atomic_json(out / "run.json", metadata)
    print(
        json.dumps(
            {
                "run": str(out),
                "device": str(device),
                "parameters": model.parameter_report(),
                "clips": len(dataset),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    if max_steps is not None and progress["updates"] >= max_steps:
        save_checkpoint(out / "last.pt", model, optimizer, scaler, progress, provenance)
        return out / "last.pt"
    while progress["epoch"] < config.epochs:
        epoch, offset = progress["epoch"], progress["offset"]
        # Loader has its own generator, so resuming does not consume the model's RNG.
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=ClipSampler(dataset, epoch, offset),
            num_workers=config.workers,
            collate_fn=collate_clips,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        group_size, pending_probes = 0, 0
        pending_policy, pending_critic = 0, 0
        records = []
        total_batches = len(loader)
        for batch_index, batch in enumerate(loader):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            warmup = min(1.0, (epoch + 1) / max(1, config.warmup_epochs))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * warmup
            amp = (
                torch.autocast("cuda", dtype=torch.float16)
                if scaler.is_enabled()
                else contextlib.nullcontext()
            )
            with amp:
                loss, metrics = training_objective(model, batch, epoch, progress["micro_steps"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective at {progress}: {metrics}")
            scaler.scale(loss / config.grad_accumulation).backward()
            group_size += 1
            pending_probes += metrics["probes"]
            pending_policy += metrics["al_policy_examples"]
            pending_critic += metrics["al_critic_examples"]
            records.append(metrics)
            progress["offset"] += batch["images"].shape[0]
            progress["micro_steps"] += 1
            boundary = group_size == config.grad_accumulation or batch_index + 1 == total_batches
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            if group_size != config.grad_accumulation:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(config.grad_accumulation / group_size)
            previous_scale = scaler.get_scale()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=not scaler.is_enabled()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                # GradScaler skips overflowing updates and backs off its scale. These
                # samples were consumed, but no policy/critic examples were optimized.
                event = dict(
                    event="amp_overflow",
                    epoch=epoch + 1,
                    micro_steps=progress["micro_steps"],
                    updates=progress["updates"],
                    scale=scaler.get_scale(),
                )
                print(json.dumps(event), flush=True)
                with (out / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(event) + "\n")
                group_size, pending_probes, pending_policy, pending_critic, records = 0, 0, 0, 0, []
                continue
            model.association.effect_examples.add_(pending_probes)
            if hasattr(model, "al"):
                model.al.policy_examples.add_(pending_policy)
                model.al.critic_examples.add_(pending_critic)
            progress["updates"] += 1
            log = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
            log.update(
                epoch=epoch + 1,
                updates=progress["updates"],
                clips_seen=progress["offset"],
                grad_norm=float(norm.detach()),
                seconds=time.monotonic() - started,
                effect_examples=int(model.association.effect_examples),
            )
            print(json.dumps(log, allow_nan=False), flush=True)
            with (out / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(log, allow_nan=False) + "\n")
            group_size, pending_probes, records = 0, 0, []
            pending_policy, pending_critic = 0, 0
            if progress["updates"] == 1 or progress["updates"] % config.checkpoint_every == 0:
                save_checkpoint(out / "last.pt", model, optimizer, scaler, progress, provenance)
            if max_steps is not None and progress["updates"] >= max_steps:
                save_checkpoint(out / "last.pt", model, optimizer, scaler, progress, provenance)
                return out / "last.pt"
        progress["epoch"] += 1
        progress["offset"] = 0
        if config.val_annotations and progress["epoch"] % config.validate_every == 0:
            from evoworld.evaluate import evaluate

            result = evaluate(
                model.eval(),
                config.val_annotations,
                config.val_images,
                out / f"validation_{progress['epoch']:03d}",
                device,
            )
            model.train()
            atomic_json(out / "validation.json", result)
            if result["idf1"] > progress["best_idf1"]:
                progress["best_idf1"] = result["idf1"]
                save_checkpoint(out / "best.pt", model, optimizer, scaler, progress, provenance)
        save_checkpoint(out / "last.pt", model, optimizer, scaler, progress, provenance)
    return out / "last.pt"

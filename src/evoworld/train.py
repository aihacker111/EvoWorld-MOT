"""Joint training on one device or torchrun DDP, with optimizer-boundary resume."""

from __future__ import annotations
import contextlib
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from evoworld.config import Config, resolve, fingerprint, atomic_json
from evoworld.data import VideoClips, ClipSampler, DistributedClipSampler, collate_clips, initialize_worker
from evoworld.distributed import DistributedContext
from evoworld.models.system import EvoWorld
from evoworld.objective import training_objective
from evoworld.runtime import (
    setup,
    environment,
    rng_state,
    restore_rng,
    save_checkpoint,
    check_resume,
    initialize_from_checkpoint,
)


def optimizer_step(model, optimizer, scaler, context, clip):
    """A nonfinite gradient on any rank skips the update on every rank."""
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip, error_if_nonfinite=not scaler.is_enabled() and not context.enabled
    )
    if context.enabled and context.any_true(~torch.isfinite(norm)):
        if not scaler.is_enabled():
            raise FloatingPointError(
                "Non-finite distributed gradients; no rank performed an optimizer update"
            )
        scaler.update(new_scale=previous_scale * scaler.get_backoff_factor())
        # update(new_scale=...) does not reset the growth tracker automatically.
        state = scaler.state_dict()
        if "_growth_tracker" in state:
            state["_growth_tracker"] = 0
            scaler.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)
        return norm, False
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return norm, not scaler.is_enabled() or scaler.get_scale() >= previous_scale


def train(config: Config, resume=None, max_steps=None, init_checkpoint=None):
    if resume and init_checkpoint:
        raise ValueError("Choose --resume or --init-checkpoint")
    context = DistributedContext.initialize(config)
    try:
        return _train(config, resume, max_steps, init_checkpoint, context)
    finally:
        context.close()


def _train(config, resume, max_steps, init_checkpoint, context):
    setup(config)
    device = context.device
    out = resolve(config.out)

    def prepare_output():
        if out.exists() and (out / "last.pt").exists() and not resume:
            raise FileExistsError(f"{out} already has a checkpoint; use --resume or a different --out")
        out.mkdir(parents=True, exist_ok=True)

    context.primary_call(prepare_output)
    dataset = VideoClips(config)
    usable = len(dataset) // context.world_size * context.world_size
    if usable == 0:
        raise ValueError("Dataset must have at least one clip per process")
    provenance = fingerprint(config)
    model = EvoWorld(config, initialize=not bool(resume or init_checkpoint)).to(device).train()
    if init_checkpoint:
        provenance["warmstart"] = initialize_from_checkpoint(model, resolve(str(init_checkpoint)))
    perception = list(model.perception.parameters())
    perception_ids = {id(p) for p in perception}
    reid_ids = {id(p) for p in model.perception.reid.parameters()}
    pretrained = [p for p in perception if id(p) not in reid_ids]
    new = [p for p in model.parameters() if id(p) not in perception_ids or id(p) in reid_ids]
    optimizer = torch.optim.AdamW(
        [
            dict(params=pretrained, lr=config.perception_lr, base_lr=config.perception_lr),
            dict(params=new, lr=config.learning_rate, base_lr=config.learning_rate),
        ],
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    progress = dict(epoch=0, offset=0, updates=0, micro_steps=0, best_idf1=-1.0)
    restored_rng = None
    if resume:
        checkpoint = torch.load(resolve(str(resume)), map_location="cpu", weights_only=False)
        saved = checkpoint.get("distributed") or {"world_size": 1}
        if saved["world_size"] != context.world_size:
            raise ValueError(
                "Resume requires the same world size; use --init-checkpoint for a new GPU topology"
            )
        if "warmstart" in checkpoint["provenance"]:
            provenance["warmstart"] = checkpoint["provenance"]["warmstart"]
        check_resume(checkpoint, config, provenance)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        progress.update(checkpoint["progress"])
        restored_rng = saved["rng_by_rank"][context.rank] if "rng_by_rank" in saved else checkpoint["rng"]
    step_model = context.wrap(model, training_objective)
    # Construction/broadcast happens before restoring each rank's stochastic stream.
    if restored_rng is not None:
        restore_rng(restored_rng)
    elif context.enabled:
        random.seed(config.seed + context.rank)
        np.random.seed(config.seed + context.rank)
        torch.manual_seed(config.seed + context.rank)
    distributed = dict(
        world_size=context.world_size,
        backend=torch.distributed.get_backend() if context.enabled else None,
        devices=context.gather_objects(str(device)),
        samples_per_epoch=usable,
        dropped_clips=len(dataset) - usable,
    )
    metadata = dict(
        config=config.dictionary(),
        provenance=provenance,
        environment=environment(),
        parameters=model.parameter_report(),
        detector_load=model.perception.load_report,
        device=str(device),
        distributed=distributed,
    )

    def publish_run():
        atomic_json(out / "run.json", metadata)
        print(
            json.dumps(
                dict(
                    run=str(out),
                    device=str(device),
                    parameters=model.parameter_report(),
                    clips=len(dataset),
                    **distributed,
                    per_gpu_batch=config.batch_size,
                    effective_batch=config.batch_size * config.grad_accumulation * context.world_size,
                )
            ),
            flush=True,
        )

    context.primary_call(publish_run)

    def checkpoint_at(path):
        topology = dict(distributed, rng_by_rank=context.gather_objects(rng_state(device)))
        context.primary_call(
            lambda: save_checkpoint(path, model, optimizer, scaler, progress, provenance, topology)
        )

    def log_record(record):
        if context.primary:
            print(json.dumps(record, allow_nan=False), flush=True)
            with (out / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record, allow_nan=False) + "\n")

    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    if max_steps is not None and progress["updates"] >= max_steps:
        checkpoint_at(out / "last.pt")
        return out / "last.pt"
    while progress["epoch"] < config.epochs:
        epoch, offset = progress["epoch"], progress["offset"]
        sampler = (
            DistributedClipSampler(dataset, epoch, context.rank, context.world_size, offset)
            if context.enabled
            else ClipSampler(dataset, epoch, offset)
        )
        generator = torch.Generator().manual_seed(config.seed + epoch + context.rank * 100003)
        options = dict(
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.workers,
            collate_fn=collate_clips,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        if context.enabled and config.workers:
            options["multiprocessing_context"] = "spawn"
            options["worker_init_fn"] = initialize_worker
        loader = DataLoader(dataset, **options)
        group_size, samples_in_group = 0, 0
        pending = [0, 0, 0]
        records = []
        for batch_index, batch in enumerate(loader):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            local_size = batch["images"].shape[0]
            boundary = group_size + 1 == config.grad_accumulation or batch_index + 1 == len(loader)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * min(1.0, (epoch + 1) / max(1, config.warmup_epochs))
            amp = (
                torch.autocast("cuda", dtype=torch.float16)
                if scaler.is_enabled()
                else contextlib.nullcontext()
            )
            synchronization = (
                step_model.no_sync() if context.enabled and not boundary else contextlib.nullcontext()
            )
            with synchronization:
                with amp:
                    # Preserve probe frequency per global microbatch; rotate the
                    # probing rank across updates instead of doubling replay work.
                    objective_step = (
                        progress["micro_steps"] * context.world_size
                        + (context.rank + progress["updates"]) % context.world_size
                    )
                    loss, metrics = step_model(batch, epoch, objective_step)
                if context.any_true(~torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite objective on at least one rank at {progress}")
                weight = local_size / (config.batch_size * config.grad_accumulation)
                scaler.scale(loss * weight).backward()
            group_size += 1
            samples_in_group += local_size
            pending = [
                a + metrics[k]
                for a, k in zip(pending, ("probes", "al_policy_examples", "al_critic_examples"))
            ]
            records.append((metrics, local_size))
            progress["offset"] += local_size * context.world_size
            progress["micro_steps"] += 1
            if not boundary:
                continue
            # Restore normalization for incomplete accumulation groups/last microbatches.
            correction = config.batch_size * config.grad_accumulation / samples_in_group
            if correction != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            norm, updated = optimizer_step(model, optimizer, scaler, context, config.grad_clip)
            if not updated:
                log_record(
                    dict(
                        event="amp_overflow",
                        epoch=epoch + 1,
                        micro_steps=progress["micro_steps"],
                        updates=progress["updates"],
                        scale=scaler.get_scale(),
                        world_size=context.world_size,
                    )
                )
                group_size, samples_in_group, pending, records = 0, 0, [0, 0, 0], []
                continue
            total_probes, total_policy, total_critic = [int(v) for v in context.sum_values(pending)]
            model.association.effect_examples.add_(total_probes)
            if hasattr(model, "al"):
                model.al.policy_examples.add_(total_policy)
                model.al.critic_examples.add_(total_critic)
            progress["updates"] += 1
            local_metrics = {
                k: sum(record[k] * size for record, size in records) / samples_in_group for k in records[0][0]
            }
            log = context.average_metrics(local_metrics)
            log.update(
                epoch=epoch + 1,
                updates=progress["updates"],
                clips_seen=progress["offset"],
                grad_norm=float(norm.detach()),
                seconds=time.monotonic() - started,
                effect_examples=int(model.association.effect_examples),
                world_size=context.world_size,
            )
            if hasattr(model, "al"):
                log.update(
                    al_policy_total=int(model.al.policy_examples),
                    al_critic_total=int(model.al.critic_examples),
                )
            log_record(log)
            group_size, samples_in_group, pending, records = 0, 0, [0, 0, 0], []
            stopping = max_steps is not None and progress["updates"] >= max_steps
            if progress["updates"] == 1 or progress["updates"] % config.checkpoint_every == 0 or stopping:
                checkpoint_at(out / "last.pt")
            if stopping:
                return out / "last.pt"
        progress["epoch"] += 1
        progress["offset"] = 0
        if config.val_annotations and progress["epoch"] % config.validate_every == 0:

            def validation():
                from evoworld.evaluate import evaluate

                try:
                    result = evaluate(
                        model.eval(),
                        config.val_annotations,
                        config.val_images,
                        out / f"validation_{progress['epoch']:03d}",
                        device,
                    )
                    atomic_json(out / "validation.json", result)
                    return result
                finally:
                    model.train()

            result = context.primary_call(validation)
            if result["idf1"] > progress["best_idf1"]:
                progress["best_idf1"] = result["idf1"]
                checkpoint_at(out / "best.pt")
        checkpoint_at(out / "last.pt")
    return out / "last.pt"

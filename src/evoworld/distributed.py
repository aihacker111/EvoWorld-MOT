"""torchrun orchestration: one process per device, global accounting and rank-local RNG."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import os
import traceback
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class TrainingStep(nn.Module):
    """DDP must own the forward that builds the entire, conditionally used loss graph."""

    def __init__(self, model, objective):
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(self, batch, epoch, step):
        return self.objective(self.model, batch, epoch, step)


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    owns_group: bool = False

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    @classmethod
    def initialize(cls, config):
        existing = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if existing else int(os.environ.get("WORLD_SIZE", "1"))
        rank = dist.get_rank() if existing else int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", str(rank)))
        wants_cuda = config.device.startswith("cuda") or (
            config.device == "auto" and torch.cuda.is_available()
        )
        if wants_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; enable GPU or explicitly use device=cpu")
            index = (
                local
                if world > 1
                else torch.device(config.device if config.device != "auto" else "cuda:0").index
            )
            index = 0 if index is None else index
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={index}, but only {torch.cuda.device_count()} CUDA devices are visible"
                )
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
        else:
            if world > 1 and config.device not in ("auto", "cpu"):
                raise ValueError("Distributed training supports CUDA or CPU/Gloo")
            device = torch.device("cpu" if config.device == "auto" else config.device)
        owns = world > 1 and not existing
        if owns:
            # Rank 0 can run a long full-sequence validation while other ranks wait.
            dist.init_process_group(backend="nccl" if wants_cuda else "gloo", timeout=timedelta(hours=3))
        return cls(device, rank, world, local, owns)

    def wrap(self, model, objective):
        step = TrainingStep(model, objective)
        if not self.enabled:
            return step
        ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(
            step, device_ids=ids, find_unused_parameters=True, broadcast_buffers=True
        )

    def any_true(self, value):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.int32).reshape(1)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def sum_values(self, values):
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.enabled:
            dist.all_reduce(tensor)
        return tensor.cpu().tolist()

    def average_metrics(self, metrics):
        keys = sorted(metrics)
        values = self.sum_values([metrics[k] for k in keys])
        return {key: value / self.world_size for key, value in zip(keys, values)}

    def gather_objects(self, value):
        if not self.enabled:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def primary_call(self, function):
        """Relay rank-0 exceptions, so a missing file or failed evaluation doesn't strand workers."""
        if not self.enabled:
            return function()
        result, error = None, None
        if self.primary:
            try:
                result = function()
            except Exception:
                error = traceback.format_exc()
        message = [(result, error)]
        if self.enabled:
            dist.broadcast_object_list(message, src=0)
        result, error = message[0]
        if error:
            raise RuntimeError(f"Rank-0 operation failed:\n{error}")
        return result

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()

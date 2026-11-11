"""Unit coverage plus opt-in actual torchrun/Gloo integration on localhost."""

from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import yaml
from evoworld.data import ClipSampler, DistributedClipSampler
from evoworld.synthetic import make_dataset
from evoworld.train import optimizer_step


def test_distributed_sampler_is_disjoint_and_resumes_at_global_offset():
    class Dataset:
        config = SimpleNamespace(seed=2026)

        def __len__(self):
            return 7

    dataset = Dataset()
    rank0 = list(DistributedClipSampler(dataset, 2, 0, 2))
    rank1 = list(DistributedClipSampler(dataset, 2, 1, 2))
    expected = list(ClipSampler(dataset, 2))[:6]
    assert len(rank0) == len(rank1) == 3
    assert not set(rank0) & set(rank1)
    assert [item for pair in zip(rank0, rank1) for item in pair] == expected
    assert list(DistributedClipSampler(dataset, 2, 0, 2, 4)) == rank0[2:]
    assert list(DistributedClipSampler(dataset, 2, 1, 2, 4)) == rank1[2:]
    with pytest.raises(ValueError, match="offset"):
        DistributedClipSampler(dataset, 2, 0, 2, 1)


def test_remote_rank_overflow_skips_local_finite_gradient_update():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    original = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    class Scaler:
        scale, tracker = 32.0, 10

        def is_enabled(self):
            return True

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.scale

        def get_backoff_factor(self):
            return 0.5

        def update(self, new_scale=None):
            self.scale = new_scale

        def state_dict(self):
            return dict(scale=self.scale, _growth_tracker=self.tracker)

        def load_state_dict(self, state):
            self.tracker = state["_growth_tracker"]

        def step(self, optimizer):
            raise AssertionError("No optimizer may step when a peer overflows")

    context = SimpleNamespace(enabled=True, any_true=lambda value: True)
    scaler = Scaler()
    norm, success = optimizer_step(model, optimizer, scaler, context, 5.0)
    assert torch.isfinite(norm) and not success
    assert scaler.scale == 16.0 and scaler.tracker == 0
    for before, after in zip(original, model.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert after.grad is None


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_TESTS") != "1" or not dist.is_available() or not dist.is_gloo_available(),
    reason="Set RUN_DISTRIBUTED_TESTS=1; requires localhost sockets",
)
def test_two_process_synchronization_and_exact_resume(config, tmp_path):
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        device="cpu",
        threads=1,
        workers=1,
        clip_step=2,
        probe_every=2,
        perception_lr=5e-5,
        reid_lr=3e-4,
        association_lr=2e-4,
        world_lr=1e-4,
        camera_lr=1e-4,
        al_lr=1e-4,
        lr_scheduler="cosine",
        min_lr_ratio=0.05,
        warmup_epochs=1,
        train_annotations=str(annotation),
        train_images=str(images),
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(root / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
    )
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if platform.system() == "Darwin" else "lo")

    def run(out, limit, resume=None):
        cfg = replace(config, out=str(out))
        path = out.with_suffix(".yaml")
        path.write_text(yaml.safe_dump(cfg.dictionary()))
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            str(root / "tests/ddp_worker.py"),
            "--config",
            str(path),
            "--steps",
            str(limit),
        ]
        if resume:
            command.extend(["--resume", str(resume)])
        completed = subprocess.run(
            command,
            env=env,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        (tmp_path / f"{out.name}-{limit}.log").write_text(completed.stdout)
        assert completed.returncode == 0, completed.stdout
        report = json.loads((out / "last.ddp.json").read_text())
        assert report["synchronized"] and report["world_size"] == 2
        return torch.load(out / "last.pt", weights_only=False)

    continuous = run(tmp_path / "continuous", 3)
    run(tmp_path / "interrupted", 1)
    resumed = run(tmp_path / "interrupted", 3, tmp_path / "interrupted/last.pt")
    assert continuous["progress"] == resumed["progress"]
    for name, tensor in continuous["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0, msg=name)
    assert continuous["distributed"]["samples_per_epoch"] == 6
    assert continuous["distributed"]["dropped_clips"] == 1
    assert len(continuous["optimizer"]["param_groups"]) == 6
    assert continuous["optimizer"]["param_groups"][0]["lr"] == pytest.approx(5e-5 * 0.525)

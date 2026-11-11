"""Regression coverage for batched mask ownership and early validation."""

from dataclasses import replace
import json
import pytest
import torch
from evoworld.models.system import EvoWorld
from evoworld.objective import training_objective
from evoworld.synthetic import make_dataset
from evoworld.train import train


@pytest.mark.parametrize("mode", ["al", "legacy"])
def test_batched_scheduled_assignment_backward(config, batch, mode):
    config = replace(config, tracker_mode=mode, teacher_forcing_start=0.0, teacher_forcing_end=0.0)
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in batch.items()}
    batch["ids"][1] += 2
    model = EvoWorld(config).train()
    loss, _ = training_objective(model, batch, epoch=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_smoke_validation_stops_early_preserves_resume_and_full_results(config, tmp_path, monkeypatch):
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        train_annotations=str(annotation),
        train_images=str(images),
        val_annotations=str(annotation),
        val_images=str(images),
        clip_step=2,
        out=str(tmp_path / "smoke"),
    )
    calls = []

    def evaluate(model, annotations, images, out, device, *, max_frames, progress):
        assert not model.training
        out.mkdir(parents=True, exist_ok=True)
        # A validation implementation may consume RNG; the next training step must be unchanged.
        torch.rand(7)
        calls.append(max_frames)
        progress(dict(frames=3, total_frames=3, sequence="synthetic"))
        return dict(idf1=0.5, mota=0.3, id_switches=1, frames=3, truncated=True)

    monkeypatch.setattr("evoworld.evaluate.evaluate", evaluate)
    output = tmp_path / "smoke"
    output.mkdir()
    sentinel = '{"full": true}'
    (output / "validation.json").write_text(sentinel)
    (output / "best.pt").write_text("full checkpoint")
    path = train(config, validate_after_steps=1, val_max_frames=3)
    first = torch.load(path, weights_only=False)
    assert first["progress"]["updates"] == 1
    assert first["progress"]["epoch"] == 0 and first["progress"]["offset"] == 2
    assert first["progress"]["best_idf1"] == -1
    path = train(config, resume=path, validate_after_steps=1, val_max_frames=3)
    second = torch.load(path, weights_only=False)
    assert second["progress"]["updates"] == 2
    assert calls == [3, 3]
    assert (output / "validation.json").read_text() == sentinel
    assert (output / "best.pt").read_text() == "full checkpoint"
    assert json.loads((output / "validation_smoke.json").read_text())["smoke_test"] is True
    baseline = train(replace(config, out=str(tmp_path / "baseline")), max_steps=2)
    baseline = torch.load(baseline, weights_only=False)
    assert second["progress"] == baseline["progress"]
    for name, value in baseline["model"].items():
        torch.testing.assert_close(second["model"][name], value, rtol=0, atol=0, msg=name)


def test_smoke_requires_validation_configuration(config):
    with pytest.raises(ValueError, match="val_annotations"):
        train(config, validate_after_steps=1)
    with pytest.raises(ValueError, match="requires --validate-after-steps"):
        train(config, val_max_frames=3)
    with pytest.raises(ValueError, match="cannot be combined"):
        train(config, max_steps=1, validate_after_steps=1)


def test_full_epoch_validation_still_runs_and_training_mode_is_restored(config, tmp_path, monkeypatch):
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        train_annotations=str(annotation),
        train_images=str(images),
        val_annotations=str(annotation),
        val_images=str(images),
        clip_step=2,
        out=str(tmp_path / "full"),
        epochs=2,
        batch_size=4,
        grad_accumulation=2,
    )
    calls = []

    def evaluate(model, annotations, images, out, device, *, max_frames, progress):
        assert not model.training and max_frames is None
        out.mkdir(parents=True, exist_ok=True)
        calls.append(out.name)
        return dict(idf1=0.5, mota=0.3, id_switches=1, frames=18, truncated=False)

    objective = training_objective

    def training(model, *args):
        assert model.training
        return objective(model, *args)

    monkeypatch.setattr("evoworld.evaluate.evaluate", evaluate)
    monkeypatch.setattr("evoworld.train.training_objective", training)
    path = train(config)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["progress"]["epoch"] == 2
    assert checkpoint["progress"]["offset"] == 0
    assert checkpoint["progress"]["best_idf1"] == 0.5
    assert calls == ["validation_001", "validation_002"]
    assert (path.parent / "best.pt").is_file()
    assert (path.parent / "validation.json").is_file()
    assert not (path.parent / "validation_smoke.json").exists()


def test_real_falcon_train_to_bounded_validation(config, tmp_path):
    from evoworld.config import resolve

    pytest.importorskip("ultralytics")
    if not resolve(config.detector_checkpoint).exists():
        pytest.skip("User detector is not bundled")
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        backend="falcon",
        train_annotations=str(annotation),
        train_images=str(images),
        val_annotations=str(annotation),
        val_images=str(images),
        clip_step=2,
        out=str(tmp_path / "falcon"),
        batch_size=2,
        grad_accumulation=1,
        teacher_forcing_start=0.0,
        teacher_forcing_end=0.0,
    )
    path = train(config, validate_after_steps=1, val_max_frames=3)
    checkpoint = torch.load(path, weights_only=False)
    result = json.loads((path.parent / "validation_smoke.json").read_text())
    assert checkpoint["progress"]["updates"] == 1
    assert checkpoint["progress"]["epoch"] == 0
    assert result["frames"] == 3 and result["available_frames"] == 18
    assert result["truncated"] is True and result["smoke_test"] is True
    assert not (path.parent / "best.pt").exists()
    assert (path.parent / "validation_smoke_u000001" / "metrics.json").is_file()
    events = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "validation_end"
    assert len([e for e in events if e.get("event") == "validation_start"]) == 1


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_DISTRIBUTED_TESTS") != "1",
    reason="Set RUN_DISTRIBUTED_TESTS=1; requires localhost sockets",
)
def test_two_process_train_to_real_falcon_validation(config, tmp_path):
    import os
    import platform
    import socket
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    from evoworld.config import resolve

    pytest.importorskip("ultralytics")
    if not resolve(config.detector_checkpoint).exists():
        pytest.skip("User detector is not bundled")
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        backend="falcon",
        train_annotations=str(annotation),
        train_images=str(images),
        val_annotations=str(annotation),
        val_images=str(images),
        clip_step=2,
        out=str(tmp_path / "ddp"),
        batch_size=2,
        grad_accumulation=1,
        workers=0,
        teacher_forcing_start=0.0,
        teacher_forcing_end=0.0,
        device="cpu",
        threads=1,
        perception_lr=5e-5,
        reid_lr=3e-4,
        association_lr=2e-4,
        world_lr=1e-4,
        camera_lr=1e-4,
        al_lr=1e-4,
        lr_scheduler="cosine",
        min_lr_ratio=0.05,
    )
    config_path = tmp_path / "ddp.yaml"
    config_path.write_text(yaml.safe_dump(config.dictionary()))
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if platform.system() == "Darwin" else "lo")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        str(root / "tools/run_training.py"),
        "--config",
        str(config_path),
        "--validate-after-steps",
        "1",
        "--val-max-frames",
        "3",
    ]
    completed = subprocess.run(
        command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
    )
    (tmp_path / "ddp.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    output = Path(config.out)
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["distributed"]["world_size"] == 2
    assert len(checkpoint["optimizer"]["param_groups"]) == 6
    assert checkpoint["progress"]["updates"] == 1 and checkpoint["progress"]["offset"] == 4
    result = json.loads((output / "validation_smoke.json").read_text())
    assert result["frames"] == 3 and result["smoke_test"] is True
    assert "Validation OK" in completed.stdout
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len([row for row in records if row.get("event") == "validation_end"]) == 1

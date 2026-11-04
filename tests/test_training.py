from dataclasses import replace
import pytest
import torch
from evoworld.config import resolve
from evoworld.models.system import EvoWorld
from evoworld.objective import training_objective
from evoworld.runtime import load_for_inference, check_resume
from evoworld.synthetic import make_dataset
from evoworld.train import train


def test_exact_optimizer_boundary_resume(config, tmp_path):
    ann, images = make_dataset(tmp_path / "data")
    config.train_annotations, config.train_images = str(ann), str(images)
    config.clip_step = 2
    config.out = str(tmp_path / "continuous")
    config.synthetic_dropout = 0.1
    continuous_path = train(config, max_steps=3)
    interrupted = replace(config, out=str(tmp_path / "interrupted"))
    interrupted_path = train(interrupted, max_steps=1)
    resumed_path = train(interrupted, resume=interrupted_path, max_steps=3)
    continuous = torch.load(continuous_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    assert continuous["progress"] == resumed["progress"]
    for name in continuous["model"]:
        torch.testing.assert_close(
            continuous["model"][name], resumed["model"][name], rtol=0, atol=0, msg=name
        )
    model, _ = load_for_inference(resumed_path)
    assert not model.training
    # A total-update limit already reached must not perform another update on resume.
    same_path = train(interrupted, resume=resumed_path, max_steps=3)
    assert torch.load(same_path, weights_only=False)["progress"]["updates"] == 3
    changed = replace(interrupted, learning_rate=0.3)
    with pytest.raises(ValueError, match="mismatch"):
        check_resume(resumed, changed, resumed["provenance"])


def test_supplied_falcon_checkpoint_full_backward_and_decode(config, batch):
    pytest.importorskip("ultralytics")
    if not resolve(config.detector_checkpoint).exists():
        pytest.skip("User-supplied detector is not bundled in this checkout")
    config.backend = "falcon"
    model = EvoWorld(config).train()
    assert model.perception.load_report["loaded_tensors"] == 493
    assert model.perception.load_report["classes"] == 10
    loss, metrics = training_objective(model, batch)
    loss.backward()
    gradient = model.perception.detector.model[0].conv.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert metrics["probes"] > 0
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert frozen == ["perception.detector.model.28.dfl.conv.weight"]
    # Curriculum decode must reuse raw pyramid outputs without corrupting detection gradients.
    frame = model.perception(batch["images"][:, 0])
    before = [value.detach().clone() for value in frame.raw]
    proposals = model.perception.training_detections(frame.raw)
    assert proposals[0]["boxes"].shape[1] == 4
    for original, value in zip(before, frame.raw):
        torch.testing.assert_close(original, value)
    # Force the actual-proposal path, including valid handling of missed detections.
    config.proposal_probability = 1.0
    config.warmup_epochs = 1
    model.zero_grad()
    curriculum_loss, _ = training_objective(model, batch, epoch=1, step=1)
    assert torch.isfinite(curriculum_loss)
    curriculum_loss.backward()
    assert model.perception.detector.model[0].conv.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        result = model.perception(batch["images"][:, 0])
        detections = model.perception.detections(result)
    assert detections[0]["boxes"].shape[1] == 4
    config.num_classes = 9
    config.category_ids = list(range(1, 10))
    with pytest.raises(ValueError, match="incompatible"):
        EvoWorld(config)


def test_v1_checkpoint_inference_and_explicit_al_warmstart(config, tmp_path):
    from evoworld.runtime import initialize_from_checkpoint
    from evoworld.tracking import EcologyTracker
    from evoworld.legacy_tracking import EcologyTracker as LegacyEcologyTracker

    old_config = replace(config, tracker_mode="legacy")
    old = EvoWorld(old_config)
    old_dictionary = {
        key: value
        for key, value in old_config.dictionary().items()
        if not key.startswith("al_") and key != "tracker_mode"
    }
    path = tmp_path / "old.pt"
    torch.save(dict(format_version=1, config=old_dictionary, model=old.state_dict()), path)
    loaded, checkpoint = load_for_inference(path)
    assert loaded.config.tracker_mode == "legacy"
    assert isinstance(EcologyTracker(loaded), LegacyEcologyTracker)
    fresh = EvoWorld(config)
    report = initialize_from_checkpoint(fresh, path)
    assert report["newly_initialized"] and all(k.startswith("al.") for k in report["newly_initialized"])
    assert fresh.al.policy_examples == 0
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], tensor, rtol=0, atol=0)
    with pytest.raises(ValueError, match="--init-checkpoint"):
        check_resume(checkpoint, config, {})


def test_overflow_skip_does_not_count_unoptimized_al_examples(config, tmp_path, monkeypatch):
    import contextlib
    from evoworld.config import resolve

    annotation, images = make_dataset(tmp_path / "data")
    config.train_annotations, config.train_images = str(annotation), str(images)
    config.out = str(tmp_path / "overflow")
    config.clip_step = 2
    config.grad_accumulation = 2

    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter):
            ctx.shape = parameter.shape
            return parameter.new_tensor(1.0)

        @staticmethod
        def backward(ctx, upstream):
            return upstream.new_full(ctx.shape, float("inf"))

    class SimulatedScaler:
        def __init__(self):
            self.value, self.bad = 32.0, False

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            self.bad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for group in optimizer.param_groups
                for p in group["params"]
            )

        def step(self, optimizer):
            if not self.bad:
                optimizer.step()

        def update(self):
            if self.bad:
                self.value /= 2

        def get_scale(self):
            return self.value

        def state_dict(self):
            return {"scale": self.value}

    calls = []

    def objective(model, batch, epoch, step):
        parameter = next(model.parameters())
        loss = BadGradient.apply(parameter) if not calls else parameter.square().mean()
        calls.append(1)
        return loss, dict(loss=float(loss.detach()), probes=3, al_policy_examples=5, al_critic_examples=7)

    monkeypatch.setattr("evoworld.train.training_objective", objective)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: SimulatedScaler())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    path = train(config, max_steps=1)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["progress"]["updates"] == 1 and len(calls) == 4
    assert checkpoint["model"]["al.policy_examples"].item() == 10
    assert checkpoint["model"]["al.critic_examples"].item() == 14
    assert checkpoint["model"]["association.effect_examples"].item() == 6
    assert "amp_overflow" in (resolve(config.out) / "metrics.jsonl").read_text()

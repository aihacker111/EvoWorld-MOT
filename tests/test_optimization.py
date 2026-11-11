"""Learning-rate ownership, actual AdamW updates and optimizer-boundary resume."""

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from evoworld.config import Config
from evoworld.models.system import EvoWorld
from evoworld.optimization import (
    build_optimizer,
    learning_rates,
    set_epoch_learning_rates,
    learning_rate_factor,
)
from evoworld.runtime import check_resume
from evoworld.synthetic import make_dataset
from evoworld.train import train


RATES = dict(perception_lr=5e-5, reid_lr=3e-4, association_lr=2e-4, world_lr=1e-4, camera_lr=1e-4, al_lr=1e-4)


def test_each_trainable_parameter_has_one_owner_and_receives_its_module_rate(config):
    config = replace(config, **RATES, weight_decay=0)
    model = EvoWorld(config)
    optimizer = build_optimizer(model, config)
    expected = {
        "detector": {
            id(p)
            for name, p in model.perception.named_parameters()
            if p.requires_grad and not name.startswith("reid.")
        },
        "reid": {id(p) for p in model.perception.reid.parameters() if p.requires_grad},
        **{
            name: {id(p) for p in getattr(model, name).parameters() if p.requires_grad}
            for name in ("world", "camera", "association", "al")
        },
    }
    seen = []
    for group in optimizer.param_groups:
        assert {id(p) for p in group["params"]} == expected[group["name"]]
        seen.extend(id(p) for p in group["params"])
        for parameter in group["params"]:
            parameter.data.zero_()
            parameter.grad = torch.ones_like(parameter)
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer.step()
    for group in optimizer.param_groups:
        expected_lr = RATES["perception_lr" if group["name"] == "detector" else group["name"] + "_lr"]
        assert group["lr"] == expected_lr
        for parameter in group["params"]:
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, -expected_lr), rtol=1e-5, atol=1e-9
            )


def test_no_overrides_preserve_original_optimizer_layout(config):
    model = EvoWorld(config)
    optimizer = build_optimizer(model, config)
    assert len(optimizer.param_groups) == 2
    detector, new = optimizer.param_groups
    assert detector["lr"] == config.perception_lr and new["lr"] == config.learning_rate
    reid_ids = {id(p) for p in model.perception.reid.parameters()}
    assert not reid_ids.intersection(id(p) for p in detector["params"])
    assert reid_ids.issubset(id(p) for p in new["params"])
    old_model = deepcopy(model)
    old_perception = list(old_model.perception.parameters())
    old_ids = {id(p) for p in old_perception}
    old_reid_ids = {id(p) for p in old_model.perception.reid.parameters()}
    old = torch.optim.AdamW(
        [
            dict(params=[p for p in old_perception if id(p) not in old_reid_ids], lr=config.perception_lr),
            dict(
                params=[p for p in old_model.parameters() if id(p) not in old_ids or id(p) in old_reid_ids],
                lr=config.learning_rate,
            ),
        ],
        weight_decay=config.weight_decay,
    )
    for left, right in zip(model.parameters(), old_model.parameters()):
        left.grad = torch.ones_like(left)
        right.grad = torch.ones_like(right)
    optimizer.step()
    old.step()
    for left, right in zip(model.parameters(), old_model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("epoch,factor", [(0, 1 / 3), (1, 2 / 3), (2, 1), (39, 1)])
def test_warmup_applies_to_every_module(config, epoch, factor):
    config = replace(config, **RATES)
    optimizer = build_optimizer(EvoWorld(config), config)
    set_epoch_learning_rates(optimizer, config, epoch)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(group["base_lr"] * factor)
    assert learning_rates(optimizer)["reid"] == pytest.approx(3e-4 * factor)


@pytest.mark.parametrize(
    "name", ["perception_lr", "learning_rate", "reid_lr", "association_lr", "world_lr", "camera_lr", "al_lr"]
)
@pytest.mark.parametrize("value", [0.0, -1e-4, float("nan"), float("inf")])
def test_invalid_learning_rates_fail_early(config, name, value):
    with pytest.raises(ValueError, match=name):
        replace(config, **{name: value}).validate()


def test_module_groups_exact_resume_and_logged_rates(config, tmp_path):
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        **RATES,
        train_annotations=str(annotation),
        train_images=str(images),
        clip_step=2,
        out=str(tmp_path / "continuous"),
    )
    continuous = torch.load(train(config, max_steps=3), weights_only=False)
    interrupted = replace(config, out=str(tmp_path / "interrupted"))
    path = train(interrupted, max_steps=1)
    resumed = torch.load(train(interrupted, resume=path, max_steps=3), weights_only=False)
    assert continuous["progress"] == resumed["progress"]
    assert len(resumed["optimizer"]["param_groups"]) == 6
    for name, tensor in continuous["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0, msg=name)
    records = [
        json.loads(line) for line in (Path(interrupted.out) / "metrics.jsonl").read_text().splitlines()
    ]
    records = [row for row in records if "loss" in row]
    assert len(records) == 3
    for row in records:
        assert row["lrs"]["detector"] == pytest.approx(5e-5 / 3)
        assert row["lrs"]["reid"] == pytest.approx(3e-4 / 3)
    metadata = json.loads((Path(interrupted.out) / "run.json").read_text())
    assert metadata["optimizer"]["base_lrs"]["detector"] == 5e-5


def test_older_config_defaults_are_expanded_before_resume_comparison(config):
    old = config.dictionary()
    old.pop("lr_scheduler")
    old.pop("min_lr_ratio")
    for key in RATES:
        if key != "perception_lr":
            old.pop(key)
    checkpoint = dict(format_version=2, config=old, provenance={})
    check_resume(checkpoint, config, {})
    with pytest.raises(ValueError, match="reid_lr"):
        check_resume(checkpoint, replace(config, reid_lr=3e-4), {})


def test_fast_profile_only_changes_optimizer_rates_and_output():
    root = Path(__file__).resolve().parents[1]
    base = Config.load(root / "configs/full.yaml").dictionary()
    fast = Config.load(root / "configs/full_fast_lr.yaml").dictionary()
    assert {k for k in base if base[k] != fast[k]} == set(RATES) | {"out"}
    for key, value in RATES.items():
        assert fast[key] == value


def test_cosine_matches_pytorch_after_warmup_and_keeps_group_ratios(config):
    config = replace(config, **RATES, epochs=40, warmup_epochs=3, lr_scheduler="cosine", min_lr_ratio=0.05)
    optimizer = build_optimizer(EvoWorld(config), config)
    golden_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=config.perception_lr)
    golden = torch.optim.lr_scheduler.CosineAnnealingLR(
        golden_optimizer,
        T_max=37,
        eta_min=config.perception_lr * 0.05,
    )
    for epoch in range(40):
        set_epoch_learning_rates(optimizer, config, epoch)
        actual = learning_rates(optimizer)
        if epoch < 3:
            assert actual["detector"] == pytest.approx(config.perception_lr * (epoch + 1) / 3)
        else:
            golden_optimizer.step()
            golden.step()
            assert actual["detector"] == pytest.approx(golden.get_last_lr()[0], rel=1e-12)
        assert actual["reid"] / actual["detector"] == pytest.approx(6.0)
        assert actual["association"] / actual["detector"] == pytest.approx(4.0)
    assert learning_rate_factor(config, 39) == pytest.approx(0.05)
    assert learning_rate_factor(config, 45) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "epochs,warmup,expected",
    [
        (3, 0, [1.0, 0.525, 0.05]),
        (1, 0, [1.0]),
        (2, 1, [1.0, 0.05]),
        (3, 3, [1 / 3, 2 / 3, 1.0]),
        (2, 3, [1 / 3, 2 / 3]),
    ],
)
def test_cosine_handles_zero_warmup_and_short_runs(config, epochs, warmup, expected):
    config = replace(config, epochs=epochs, warmup_epochs=warmup, lr_scheduler="cosine", min_lr_ratio=0.05)
    assert [learning_rate_factor(config, e) for e in range(epochs)] == pytest.approx(expected)
    with pytest.raises(ValueError, match="nonnegative"):
        learning_rate_factor(config, -1)


@pytest.mark.parametrize(
    "option,value",
    [
        ("lr_scheduler", "cosin"),
        ("min_lr_ratio", -0.1),
        ("min_lr_ratio", 1.1),
        ("min_lr_ratio", float("nan")),
        ("min_lr_ratio", float("inf")),
    ],
)
def test_invalid_scheduler_settings_fail_early(config, option, value):
    with pytest.raises(ValueError, match=option):
        replace(config, **{option: value}).validate()


def test_cosine_mid_epoch_resume_keeps_exact_lr_and_model(config, tmp_path):
    annotation, images = make_dataset(tmp_path / "data")
    config = replace(
        config,
        **RATES,
        train_annotations=str(annotation),
        train_images=str(images),
        clip_step=2,
        epochs=4,
        warmup_epochs=1,
        lr_scheduler="cosine",
        min_lr_ratio=0.05,
        out=str(tmp_path / "continuous"),
    )
    continuous = torch.load(train(config, max_steps=6), weights_only=False)
    interrupted = replace(config, out=str(tmp_path / "interrupted"))
    path = train(interrupted, max_steps=5)
    before = torch.load(path, weights_only=False)
    assert before["progress"]["epoch"] == 1 and before["progress"]["offset"] == 2
    saved_lr = before["optimizer"]["param_groups"][0]["lr"]
    assert saved_lr == pytest.approx(RATES["perception_lr"] * 0.7625)
    resumed = torch.load(train(interrupted, resume=path, max_steps=6), weights_only=False)
    assert continuous["progress"] == resumed["progress"]
    for name, tensor in continuous["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0, msg=name)
    for actual, expected in zip(
        resumed["optimizer"]["param_groups"], continuous["optimizer"]["param_groups"]
    ):
        assert actual == expected
    records = [
        json.loads(line) for line in (Path(interrupted.out) / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["lrs"]["detector"] for row in records if "loss" in row][-2:] == pytest.approx([saved_lr] * 2)
    for option, value in (("lr_scheduler", "constant"), ("min_lr_ratio", 0.1)):
        with pytest.raises(ValueError, match=option):
            check_resume(resumed, replace(interrupted, **{option: value}), resumed["provenance"])

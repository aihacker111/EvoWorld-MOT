import pytest
import torch
from evoworld.config import Config
from evoworld.geometry import identity_affine


@pytest.fixture
def config():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return Config(
        backend="tiny",
        width=96,
        height=64,
        embedding_dim=16,
        hidden_dim=32,
        stochastic_dim=4,
        camera_dim=16,
        clip_length=6,
        burn_in=1,
        horizon=2,
        probe_horizon=2,
        workers=0,
        probe_every=1,
        effect_min_examples=1,
        grad_accumulation=2,
        epochs=3,
    ).validate()


@pytest.fixture
def batch(config):
    boxes = torch.tensor([0.3, 0.4, 0.12, 0.2, 0.7, 0.4, 0.12, 0.2]).reshape(1, 1, 2, 4).repeat(1, 6, 1, 1)
    boxes[:, :, 0, 0] += torch.arange(6) * 0.01
    return dict(
        images=torch.randn(1, 6, 3, 64, 96),
        boxes=boxes,
        present=torch.ones(1, 6, 2, dtype=torch.bool),
        classes=torch.zeros(1, 2, dtype=torch.long),
        ids=torch.tensor([[0, 1]]),
        affine=identity_affine(boxes, 1, 6),
        camera_valid=torch.ones(1, 6, dtype=torch.bool),
        delta=torch.ones(1, 6),
    )

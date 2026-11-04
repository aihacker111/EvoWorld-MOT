import torch
import torch.nn.functional as F
from evoworld.geometry import warp_boxes, identity_affine
from evoworld.models.system import EvoWorld
from evoworld.models.association import assign, blend_memory
from evoworld.objective import training_objective
from evoworld.interventions import replay


def test_affine_envelope_and_modes():
    box = torch.tensor([[[0.5, 0.4, 0.2, 0.1]]])
    transform = torch.tensor([[[0.0, -1.0, 1.0], [1.0, 0.0, 0.1]]])
    expected = torch.tensor([[[0.6, 0.6, 0.1, 0.2]]])
    torch.testing.assert_close(warp_boxes(box, transform), expected)
    modes = box.unsqueeze(2).repeat(1, 1, 3, 1)
    torch.testing.assert_close(warp_boxes(modes, transform), expected.unsqueeze(2).expand_as(modes))


def test_assignment_null_and_one_to_one():
    scores = torch.tensor([[3.0, 2.0], [2.9, -10.0], [-100.0, -100.0]])
    pairs = assign(scores, -1.0)
    assert set(pairs) == {(0, 1), (1, 0)}
    assert len({j for _, j in pairs}) == len(pairs)
    assert assign(scores, -1.0, (1, 0)) != pairs
    assert assign(torch.empty(0, 3), -1) == []


def test_zero_write_never_invents_measurements():
    memory = F.normalize(torch.randn(2, 8), dim=-1)
    observation = F.normalize(torch.randn(2, 8), dim=-1)
    torch.testing.assert_close(blend_memory(memory, observation, torch.zeros(2)), memory)
    torch.testing.assert_close(blend_memory(memory, observation, torch.ones(2)), observation)


def test_full_objective_gradients_and_empty_frames(config, batch):
    model = EvoWorld(config).train()
    model.association.effect_examples.fill_(20)
    loss, metrics = training_objective(model, batch)
    assert torch.isfinite(loss) and metrics["probes"] > 0
    loss.backward()
    for name, module in model.named_children():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients, name
        assert all(torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients) > 0, name
    assert model.world.acceleration[-1].weight.grad.abs().sum() > 0
    assert model.association.write[-1].weight.grad.abs().sum() > 0
    batch["present"].zero_()
    model.zero_grad()
    loss, _ = training_objective(model, batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_counterfactual_replay_isolated_and_reproducible(config, batch):
    model = EvoWorld(config).eval()
    box = batch["boxes"][:, 0]
    feature = F.normalize(torch.randn(1, 2, config.embedding_dim), dim=-1)
    state = model.world.initial(box, 1, 2)
    state, _ = model.world.correct(state, box, feature, batch["present"][:, 0], torch.ones(1))
    original = state.clone()
    observations = [(box[0], feature[0], batch["present"][0, 0], identity_affine(box, 1), torch.ones(1))] * 2
    good = replay(
        model,
        state,
        feature[0],
        feature[0],
        batch["classes"][0],
        observations,
        [(0, 0), (1, 1)],
        torch.tensor(0.25),
        0,
    )
    wrong = replay(
        model,
        state,
        feature[0],
        feature[0],
        batch["classes"][0],
        observations,
        [(0, 1), (1, 0)],
        torch.tensor(1.0),
        0,
    )
    repeated = replay(
        model,
        state,
        feature[0],
        feature[0],
        batch["classes"][0],
        observations,
        [(0, 0), (1, 1)],
        torch.tensor(0.25),
        0,
    )
    torch.testing.assert_close(good[2], repeated[2])
    torch.testing.assert_close(state.h, original.h)
    torch.testing.assert_close(state.box, original.box)
    assert wrong[0] >= 1
    assert not torch.equal(good[2], wrong[2])

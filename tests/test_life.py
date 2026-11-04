import torch
import torch.nn.functional as F
from evoworld.models.system import EvoWorld
from evoworld.models.life import Phase
from evoworld.al_tracking import LifeTracker
from evoworld.al_learning import memory_supervision
from evoworld.geometry import identity_affine
from evoworld.planning import imagine, pooled_features, ComputeBudget
from evoworld.models.association import assign_with_null


def data(config, boxes):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return dict(
        boxes=boxes, scores=torch.full((len(boxes),), 0.9), classes=torch.zeros(len(boxes), dtype=torch.long)
    )


def inputs(config):
    return torch.ones(1, config.embedding_dim, 16, 24), torch.zeros(1, 16)


def test_agent_occlusion_uncertainty_and_identity_recovery(config):
    model = EvoWorld(config).eval()
    tracker = LifeTracker(model)
    dense, camera = inputs(config)
    obs = data(config, [[0.4, 0.4, 0.15, 0.2]])
    assert tracker.step(obs, dense, camera, 1) == []  # confirmation requires real repeated evidence
    visible = tracker.step(obs, dense, camera, 2)
    identity = visible[0][0]
    before, memory = tracker.agents.variance.clone(), tracker.memory.clone()
    for frame in range(3, 7):
        assert tracker.step(data(config, []), dense, camera, frame) == []
        assert tracker.agents.phase[0] == Phase.DORMANT
        torch.testing.assert_close(tracker.memory, memory, atol=0, rtol=0)
    assert (tracker.agents.variance > before).all()
    recovered = tracker.step(obs, dense, camera, 7)
    assert recovered[0][0] == identity
    assert tracker.last_diagnostics["recovered_ids"] == [identity]
    assert tracker.agents.phase[0] == Phase.ACTIVE
    assert (tracker.agents.variance < before + 0.001).all()


def test_unconfirmed_false_positive_expires_without_ghost_output(config):
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    assert tracker.step(data(config, [[0.4, 0.4, 0.1, 0.2]]), dense, camera, 1) == []
    identity = int(tracker.ids[0])
    for frame in range(2, 2 + config.al_tentative_age + 1):
        assert tracker.step(data(config, []), dense, camera, frame) == []
    assert len(tracker.ids) == 0
    assert tracker.last_diagnostics["terminated_ids"] == [identity]


def test_learned_termination_and_observation_protection(config):
    model = EvoWorld(config).eval()
    tracker = LifeTracker(model)
    dense, camera = inputs(config)
    obs = data(config, [[0.4, 0.4, 0.1, 0.2]])
    tracker.step(obs, dense, camera, 1)
    for frame in range(2, 5):
        assert len(tracker.step(obs, dense, camera, frame)) == 1
        assert len(tracker.ids) == 1  # evidence prevents a learned early-death error
    model.al.policy_examples.fill_(config.al_min_examples)
    with torch.no_grad():
        model.al.existence.bias.fill_(-20)
        model.al.lifecycle.bias.copy_(torch.tensor([-20.0, -20.0, 20.0]))
    identity = int(tracker.ids[0])
    for frame in range(5, 20):
        tracker.step(data(config, []), dense, camera, frame)
        if len(tracker.ids) == 0:
            break
    assert frame < config.al_max_dormant_age
    assert tracker.last_diagnostics["terminated_ids"] == [identity]


def test_global_arbitration_never_duplicates_detection(config):
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    two = data(config, [[0.4, 0.4, 0.2, 0.2], [0.6, 0.4, 0.2, 0.2]])
    tracker.step(two, dense, camera, 1)
    tracker.step(two, dense, camera, 2)
    rows = tracker.step(data(config, [[0.5, 0.4, 0.2, 0.2]]), dense, camera, 3)
    assert len(rows) == 1
    assert len(tracker.ids) == 2  # losing agent waits rather than stealing the measurement
    assert int((tracker.agents.phase == Phase.DORMANT).sum()) == 1
    pairs = assign_with_null(torch.tensor([[2.0], [3.0]]), torch.tensor([0.0, 4.0]))
    assert pairs == [(0, 0)]


def test_budget_counts_actual_forecasts_and_roi_rechecks(config):
    config.al_compute_budget = 2
    model = EvoWorld(config).eval()
    model.al.policy_examples.fill_(config.al_min_examples)
    with torch.no_grad():
        model.al.resource.bias.fill_(10)
    tracker = LifeTracker(model)
    dense, camera = inputs(config)
    obs = data(config, [[0.3, 0.4, 0.1, 0.2], [0.7, 0.4, 0.1, 0.2]])
    tracker.step(obs, dense, camera, 1)
    tracker.step(obs, dense, camera, 2)
    report = tracker.last_diagnostics
    assert report["forecast_agent_steps"] == 0  # each rollout needs three tokens per agent
    assert report["roi_rechecks"] == 2 and report["compute_tokens"] == 2
    assert (tracker.agents.credit >= 0).all()
    config.al_compute_budget = 0
    tracker.step(obs, dense, camera, 3)
    assert tracker.last_diagnostics["compute_tokens"] == 0


def test_forecast_is_shared_across_memory_actions(config, monkeypatch):
    model = EvoWorld(config).eval()
    box = torch.tensor([[[0.4, 0.4, 0.1, 0.2], [0.7, 0.4, 0.1, 0.2]]])
    state = model.world.initial(box, 1, 2)
    state.box, state.valid = box, torch.ones(1, 2, dtype=torch.bool)
    calls = []
    original = model.world.predict

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.world, "predict", spy)
    cache = imagine(model, state, identity_affine(box, 1), torch.tensor([0, 1]), 3)
    before = cache.boxes.clone()
    memory = F.normalize(torch.randn(2, config.embedding_dim), dim=-1)
    measured = F.normalize(torch.randn_like(memory), dim=-1)
    for value in (0.0, 0.2, 1.0):
        cache.affinities(memory, measured, torch.full((2,), value), torch.arange(2))
    assert len(calls) == 3
    torch.testing.assert_close(cache.boxes, before, rtol=0, atol=0)
    assert not cache.boxes.requires_grad


def test_repair_gate_learns_and_is_executed(config):
    config.al_forecast_weight = 0.0
    model = EvoWorld(config)
    h = torch.zeros(1, config.hidden_dim)
    truth = F.normalize(torch.ones(1, config.embedding_dim), dim=-1)
    memory = -truth
    box = torch.tensor([[0.4, 0.4, 0.1, 0.2]])
    context = model.association.context(
        h, memory, truth, truth, box, box, torch.ones(1), torch.ones(1), torch.zeros(1)
    )
    prior = model.association.gate(context)
    gate, loss = memory_supervision(
        model,
        context,
        torch.zeros(1, model.al.statistics_dim),
        torch.zeros(1, 4),
        prior,
        memory,
        truth,
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )
    loss.backward()
    assert (
        model.al.candidate[-1].bias.grad[1] < 0
    )  # gradient raises write gate on a correct repair observation
    model.eval()
    model.al.policy_examples.fill_(config.al_min_examples)
    with torch.no_grad():
        model.al.candidate[-1].bias[1] = 8.0
    tracker = LifeTracker(model)
    dense, camera = inputs(config)
    obs = data(config, [[0.4, 0.4, 0.1, 0.2]])
    tracker.step(obs, dense, camera, 1)
    tracker.step(obs, dense, camera, 2)
    tracker.memory.neg_()  # corrupt working memory; preserve the protected anchor
    tracker.step(obs, dense, camera, 3)
    assert (tracker.memory * tracker.anchor).sum(-1).item() > 0.9


def test_roi_samples_real_image_and_budget_refuses_overdraft(config):
    dense = torch.randn(1, config.embedding_dim, 16, 24)
    boxes = torch.tensor([[[0.3, 0.4, 0.1, 0.2]]])
    actual = pooled_features(dense, boxes)
    assert actual.shape == (1, 1, config.embedding_dim)
    assert torch.isfinite(actual).all()
    budget = ComputeBudget(2)
    assert budget.spend(2, "roi")
    assert not budget.spend(1, "forecast")
    assert budget.spent == 2


def test_learned_birth_rejects_false_detection(config):
    model = EvoWorld(config).eval()
    model.al.policy_examples.fill_(config.al_min_examples)
    with torch.no_grad():
        model.al.existence.bias.fill_(-20)
    tracker = LifeTracker(model)
    dense, camera = inputs(config)
    assert tracker.step(data(config, [[0.4, 0.4, 0.1, 0.2]]), dense, camera, 1) == []
    assert len(tracker.ids) == 0


def test_actor_gradient_does_not_update_its_critic(config, batch):
    from evoworld.interventions import intervention_loss
    from evoworld.al_learning import effect_supervision
    from evoworld.models.life import AgentState

    config.al_burst_probability = 0.0
    model = EvoWorld(config).train()
    model.al.critic_examples.fill_(config.al_critic_min_examples)
    boxes = batch["boxes"][:, 0]
    feature = F.normalize(torch.randn(1, 2, config.embedding_dim), dim=-1)
    state = model.world.initial(boxes, 1, 2)
    state, _ = model.world.correct(state, boxes, feature, batch["present"][:, 0], torch.ones(1))
    observations = [
        (boxes[0], feature[0], batch["present"][0, 0], identity_affine(boxes, 1), torch.ones(1))
    ] * 2
    samples = intervention_loss(model, state, feature[0], batch["classes"][0], observations, collect=True)[3]
    _, actor, count = effect_supervision(
        model,
        samples,
        AgentState.initial(boxes, 2),
        feature[0],
        batch["classes"][0],
        identity_affine(boxes, 1),
        observations,
    )
    assert count > 0
    actor.backward()
    assert all(p.grad is None for p in model.al.critic.parameters())
    assert model.al.candidate[-1].weight.grad.abs().sum() > 0


def test_dropout_and_corruption_preserve_ground_truth_and_anchor(config):
    from evoworld.al_learning import observation_masks, corrupt_memory

    config.synthetic_dropout = 0.0
    config.al_burst_probability = 1.0
    truth = torch.ones(1, 6, 2, dtype=torch.bool)
    observations = observation_masks(truth, config)
    assert truth.all() and not observations.all()
    memory = F.normalize(torch.randn(1, 2, config.embedding_dim), dim=-1)
    anchor = memory.clone()
    corrupted = corrupt_memory(memory, torch.ones(1, 2, dtype=torch.bool), 1.0)
    torch.testing.assert_close(memory, anchor, rtol=0, atol=0)
    assert not torch.equal(corrupted, anchor)
    torch.testing.assert_close(corrupted.norm(dim=-1), torch.ones(1, 2))

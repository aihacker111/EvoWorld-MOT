"""v0.3 train/inference alignment: class groups, false positives, bounded dormancy, loaders."""

from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from evoworld.classes import class_compatibility, group_nms
from evoworld.models.system import EvoWorld
from evoworld.models.life import AgentState, Phase
from evoworld.models.world import damp_missing
from evoworld.al_tracking import LifeTracker
from evoworld.objective import training_objective, link_false_tracks, proposal_schedule
from evoworld.al_learning import ghost_boxes, observation_masks
from evoworld.optimization import build_optimizer, learning_rates
from evoworld.runtime import transfer_state
from evoworld.evaluate import backfill_delay


def detections(boxes, scores=None, classes=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    n = len(boxes)
    return dict(
        boxes=boxes,
        scores=torch.full((n,), 0.9) if scores is None else torch.tensor(scores, dtype=torch.float32),
        classes=torch.zeros(n, dtype=torch.long) if classes is None else torch.tensor(classes),
    )


def inputs(config):
    return torch.ones(1, config.embedding_dim, 16, 24), torch.zeros(1, 16)


def test_class_groups_soft_within_and_hard_across(config):
    tracks = torch.tensor([3, 3, 0])  # car, car, pedestrian
    dets = torch.tensor([4, 0])  # van, pedestrian
    allowed, penalty = class_compatibility(tracks, dets, config)
    assert allowed.tolist() == [[True, False], [True, False], [False, True]]
    assert penalty[0, 0] == config.class_mismatch_penalty and penalty[2, 1] == 0
    strict = replace(config, class_groups=[])
    allowed, _ = class_compatibility(tracks, dets, strict)
    assert not allowed[0, 0]


def test_group_nms_removes_cross_class_duplicates_only_inside_group(config):
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    classes = torch.tensor([3, 4, 0])  # car, van (same group), pedestrian (other group)
    keep = group_nms(boxes, scores, classes, config)
    assert keep.tolist() == [0, 2]


def test_scores_gate_far_detections_and_new_confidence_input_starts_inert(config):
    model = EvoWorld(config).eval()
    box = torch.tensor([[[0.3, 0.3, 0.05, 0.05]]])
    state = model.world.initial(box, 1, 1)
    state.box = box
    memory = F.normalize(torch.ones(1, config.embedding_dim), dim=-1)
    near_far = torch.tensor([[0.31, 0.3, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]])
    features = memory.expand(2, -1)
    classes = torch.zeros(1, dtype=torch.long)
    scores = model.association.scores(state, memory, memory, near_far, features, classes, classes.expand(2))
    assert torch.isfinite(scores[0, 0]) and torch.isinf(scores[0, 1])
    low = model.association.scores(
        state, memory, memory, near_far, features, classes, classes.expand(2), confidence=torch.full((2,), 0.1)
    )
    torch.testing.assert_close(scores[0, 0], low[0, 0])  # zero-initialized learned term


def test_missing_uncertainty_is_bounded_and_velocity_decays(config):
    agents = AgentState.initial(torch.zeros(1), 1)
    forecast = dict(log_sigma=torch.full((1, 1, 4), -0.5))
    box = torch.tensor([[0.5, 0.5, 0.02, 0.02]])
    for _ in range(200):
        agents = agents.predict(forecast, torch.ones(1), config, box)
    cap = (config.dormant_std_cap * 0.02 + 0.01) ** 2
    assert float(agents.variance.max()) <= cap + 1e-6
    model = EvoWorld(config)
    state = model.world.initial(box[None], 1, 2)
    state.valid[:] = True
    state.velocity[:] = 0.1
    damped = damp_missing(state, torch.tensor([[True, False]]), torch.tensor([2.0]), 0.9)
    torch.testing.assert_close(damped.velocity[0, 0], torch.full((2,), 0.1))
    torch.testing.assert_close(damped.velocity[0, 1], torch.full((2,), 0.1 * 0.81))


def test_statistics_saturate_inside_unit_range_for_long_lived_agents(config):
    agents = AgentState.initial(torch.zeros(1), 2)
    agents.age[:] = torch.tensor([1, 10_000])
    agents.hits[:] = torch.tensor([1, 10_000])
    agents.lifetime[:] = torch.tensor([1, 10_000])
    model = EvoWorld(config)
    state = model.world.initial(torch.zeros(1), 1, 2)
    forecast = dict(visibility=torch.zeros(1, 2))
    stats = agents.statistics(state, forecast, torch.zeros(2, 0), config)
    assert ((stats[:, 3:6] >= 0) & (stats[:, 3:6] <= 1)).all()
    assert (stats[0, 3:6] < stats[1, 3:6]).all()


def test_class_flip_inside_group_keeps_identity_and_votes_majority(config):
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    box = [[0.4, 0.4, 0.1, 0.1]]
    tracker.step(detections(box, classes=[3]), dense, camera, 1)
    identity = tracker.step(detections(box, classes=[3]), dense, camera, 2)[0][0]
    rows = tracker.step(detections(box, classes=[4]), dense, camera, 3)  # detector says van
    assert rows[0][0] == identity and rows[0][4] == 3  # majority class remains car
    assert len(tracker.ids) == 1


def test_duplicate_of_matched_agent_is_not_born(config):
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    box = [[0.4, 0.4, 0.1, 0.1]]
    tracker.step(detections(box), dense, camera, 1)
    tracker.step(detections(box), dense, camera, 2)
    twin = detections([[0.4, 0.4, 0.1, 0.1], [0.401, 0.4, 0.1, 0.1]], classes=[0, 1])
    tracker.step(twin, dense, camera, 3)
    assert len(tracker.ids) == 1


def test_low_confidence_box_is_claimed_by_recent_agent_without_memory_write(config):
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    box = [[0.4, 0.4, 0.1, 0.1]]
    tracker.step(detections(box), dense, camera, 1)
    identity = tracker.step(detections(box), dense, camera, 2)[0][0]
    memory = tracker.memory.clone()
    rows = tracker.step(detections(box, scores=[0.2]), dense, camera, 3)
    assert [r[0] for r in rows] == [identity]
    assert tracker.agents.phase[0] == Phase.ACTIVE
    torch.testing.assert_close(tracker.memory, memory)


def test_backfill_releases_tentative_frames_on_confirmation(config):
    config.backfill_tentative = True
    tracker = LifeTracker(EvoWorld(config).eval())
    dense, camera = inputs(config)
    box = [[0.4, 0.4, 0.1, 0.1]]
    assert tracker.step(detections(box), dense, camera, 1) == []
    rows = tracker.step(detections(box), dense, camera, 2)
    assert [(f, i) for f, i, *_ in tracker.last_backfill] == [(1, rows[0][0])]
    assert backfill_delay(config) >= 1


def fake_proposals(model, config, extra_far=True):
    """Detector stand-in: one localization of the first GT object plus false positives."""

    def training_detections(raw):
        boxes = [[0.31, 0.4, 0.12, 0.2], [0.1, 0.85, 0.05, 0.05], [0.9, 0.1, 0.05, 0.05]]
        return [
            dict(
                boxes=torch.tensor(boxes),
                scores=torch.tensor([0.9, 0.8, 0.3]),
                classes=torch.tensor([1, 0, 0]),  # people/pedestrian confusion inside one group
            )
        ]

    model.perception.training_detections = training_detections


def test_objective_with_false_positive_distractors_and_tracks(config, batch):
    config.proposal_probability = 1.0
    config.warmup_epochs = 1
    config.legacy_effect_in_al = False
    config.on_policy_replay = True
    config.rollout_decay = 0.8
    config.random_agent_history = 0.5
    config.perception_grad_every = 2
    config.al_burst_count = 2
    model = EvoWorld(config).train()
    model.al.critic_examples.fill_(config.al_critic_min_examples)
    fake_proposals(model, config)
    loss, metrics = training_objective(model, batch, epoch=1, step=0)
    assert torch.isfinite(loss)
    assert metrics["distractors"] > 0 and metrics["false_tracks"] >= 1
    assert metrics["effect"] == 0 and metrics["stability"] == 0 and metrics["gate"] == 0
    loss.backward()
    assert model.association.score_weights.grad is not None
    assert model.association.effect[0].weight.grad is None  # v1 critic is off in AL runs
    for p in model.al.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_false_track_linking_and_ghost_sizes(config):
    boxes = torch.tensor([[0.1, 0.1, 0.05, 0.05]])
    frames = [
        (boxes, torch.zeros(1, 4), torch.tensor([0.9]), torch.tensor([0])),
        (boxes + 0.005, torch.zeros(1, 4), torch.tensor([0.8]), torch.tensor([1])),
        (boxes.new_zeros(0, 4), torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)),
    ]
    tracks = link_false_tracks(frames, config, 4)
    assert len(tracks) == 1 and tracks[0]["frames"] == {0: 0, 1: 0}
    gt = torch.tensor([[0.5, 0.5, 0.02, 0.03]])
    ghosts = ghost_boxes(gt, torch.tensor([True]), 4)
    assert len(ghosts) == 4
    assert (ghosts[:, 2:] <= 0.04).all()  # sized like real objects, not a fixed large crop


def test_bursts_and_proposal_curriculum(config):
    config.synthetic_dropout = 0.0
    config.al_burst_probability = 1.0
    config.al_burst_length = 4
    masks = observation_masks(torch.ones(4, 6, 3, dtype=torch.bool), config)
    assert not masks[:, 1:].all()
    config.epochs, config.warmup_epochs = 10, 2
    config.proposal_probability, config.proposal_probability_end = 0.5, 0.9
    assert proposal_schedule(config, 0) == 0
    assert proposal_schedule(config, 2) == pytest.approx(0.5)
    assert proposal_schedule(config, 9) == pytest.approx(0.9)


def test_old_checkpoint_transfers_with_identical_function(config):
    model = EvoWorld(config).eval()
    with torch.no_grad():
        model.association.pair[-1].weight.normal_()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["association.pair.0.weight"] = state["association.pair.0.weight"][:, :-1].clone()
    state.pop("association.score_weights")
    fresh = EvoWorld(config).eval()
    report = transfer_state(fresh, state)
    assert report["expanded_inputs"] == ["association.pair.0.weight"]
    assert report["newly_initialized"] == ["association.score_weights"]
    box = torch.tensor([[[0.3, 0.3, 0.1, 0.1]]])
    s = model.world.initial(box, 1, 1)
    s.box = box
    memory = F.normalize(torch.randn(1, config.embedding_dim), dim=-1)
    classes = torch.zeros(1, dtype=torch.long)
    old = state["association.pair.0.weight"]
    assert torch.equal(fresh.association.pair[0].weight[:, : old.shape[1]], old)
    assert fresh.association.pair[0].weight[:, -1].abs().sum() == 0
    a = fresh.association.scores(s, memory, memory, box[0], memory, classes, classes)
    b = fresh.association.scores(s, memory, memory, box[0], memory, classes, classes, confidence=torch.zeros(1))
    torch.testing.assert_close(a, b)
    broken = dict(state, **{"association.write.0.weight": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="mismatch"):
        transfer_state(EvoWorld(config), broken)


def test_no_decay_groups_cover_every_parameter_once(config):
    config.no_decay_norm_bias = True
    model = EvoWorld(config)
    optimizer = build_optimizer(model, config)
    seen = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == len([p for p in model.parameters() if p.requires_grad])
    for group in optimizer.param_groups:
        if group["name"].endswith("/no_decay"):
            assert group["weight_decay"] == 0 and all(p.ndim <= 1 for p in group["params"])
    assert not any(name.endswith("/no_decay") for name in learning_rates(optimizer))


def test_multiple_strides_add_long_gap_windows(config, tmp_path):
    from evoworld.data import VideoClips
    from evoworld.synthetic import make_dataset

    annotation, images = make_dataset(tmp_path / "data")
    config = replace(config, train_annotations=str(annotation), train_images=str(images), clip_step=2)
    base = VideoClips(config)
    multi = VideoClips(replace(config, frame_stride_choices=[1, 2]))
    assert len(multi) > len(base)
    gaps = {int(d) for item in (multi[i] for i in range(len(multi))) for d in item["delta"][1:]}
    assert gaps == {1, 2}

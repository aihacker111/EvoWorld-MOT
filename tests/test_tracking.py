import pytest
import torch
from evoworld.models.system import EvoWorld
from evoworld.tracking import EcologyTracker


def observations(config, boxes, classes=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return dict(
        boxes=boxes,
        scores=torch.ones(len(boxes)) * 0.9,
        classes=torch.zeros(len(boxes), dtype=torch.long) if classes is None else torch.tensor(classes),
    )


def test_tracking_births_misses_classes_and_reset(config):
    config.tracker_mode = "legacy"
    model = EvoWorld(config).eval()
    tracker = EcologyTracker(model)
    dense = torch.randn(1, config.embedding_dim, 16, 24)
    camera = torch.zeros(1, 16)
    d = observations(config, [[0.3, 0.4, 0.1, 0.2], [0.7, 0.4, 0.1, 0.2]])
    first = tracker.step(d, dense, camera, 1)
    assert len(first) == 2 and len({r[0] for r in first}) == 2
    second = tracker.step(d, dense, camera, 2)
    assert [r[0] for r in first] == [r[0] for r in second]
    assert tracker.step(observations(config, []), dense, camera, 3) == []
    for h in tracker.population:
        assert len(set(h.ids.tolist())) == len(h.ids)
        # Missing measurements do not overwrite appearance memory.
        assert torch.isfinite(h.memory).all()
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.step(d, dense, camera, 3)
    tracker.reset()
    assert len(tracker.step(d, dense, camera, 1)) == 2
    mismatched = observations(config, [[0.3, 0.4, 0.1, 0.2]], [2])
    old_id = tracker.population[0].ids[0]
    rows = tracker.step(mismatched, dense, camera, 2)
    assert all(r[0] != old_id for r in rows)


def test_ecology_retains_bounded_distinct_descendants(config):
    config.tracker_mode = "legacy"
    config.uncertain_margin = 100
    config.ecology_budget = 4
    tracker = EcologyTracker(EvoWorld(config).eval())
    dense = torch.ones(1, config.embedding_dim, 16, 24)
    camera = torch.zeros(1, 16)
    d = observations(config, [[0.48, 0.4, 0.2, 0.2], [0.52, 0.4, 0.2, 0.2]])
    tracker.step(d, dense, camera, 1)
    tracker.step(d, dense, camera, 2)
    assert 1 < len(tracker.population) <= config.ecology_budget
    for h in tracker.population:
        assert len({row[1] for row in h.output}) == len(h.output)
    tracker.step(d, dense, camera, 3)
    assert 1 < len(tracker.population) <= config.ecology_budget
    assert any(len(h.lineage) >= 3 for h in tracker.population)


def test_preview_falls_back_when_mp4_encoder_is_missing(monkeypatch, tmp_path):
    from evoworld.inference import open_preview

    attempted = []

    class Writer:
        def __init__(self, path, *args):
            self.path = path
            self.released = False
            attempted.append(self)

        def isOpened(self):
            return self.path.endswith(".avi")

        def release(self):
            self.released = True

    monkeypatch.setattr("evoworld.inference.cv2.VideoWriter", Writer)
    writer, path = open_preview(tmp_path, 25.0, 96, 64)
    assert path.name == "preview.avi"
    assert attempted[0].released and writer.isOpened()

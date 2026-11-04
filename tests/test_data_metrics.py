import json
import numpy as np
import pytest
import torch
from evoworld.config import Config
from evoworld.data import CocoVideo, VideoClips, collate_clips, ClipSampler, prepare_image
from evoworld.synthetic import make_dataset
from evoworld.evaluate import IdentityMetrics, summarize, ignore_unmatched


def test_ordered_clips_and_deterministic_epoch_augmentation(config, tmp_path):
    ann, root = make_dataset(tmp_path)
    config.train_annotations, config.train_images = str(ann), str(root)
    dataset = VideoClips(config)
    first = dataset[(4, 0)]
    repeated = dataset[(4, 0)]
    torch.testing.assert_close(first["images"], repeated["images"])
    torch.testing.assert_close(first["affine"], repeated["affine"])
    assert first["frame_ids"] == sorted(first["frame_ids"])
    assert len(set(first["ids"].tolist())) == 2
    batch = collate_clips([first, repeated])
    assert batch["boxes"].shape == (2, config.clip_length, 2, 4)
    keys = list(ClipSampler(dataset, 3))
    assert list(ClipSampler(dataset, 3, 2)) == keys[2:]


def test_duplicate_identity_and_missing_temporal_metadata_rejected(config, tmp_path):
    ann, root = make_dataset(tmp_path)
    payload = json.loads(ann.read_text())
    payload["annotations"].append(dict(payload["annotations"][0], id=999))
    ann.write_text(json.dumps(payload))
    data = CocoVideo(ann, root, config)
    frame = data.sequences["crossing"][0]
    with pytest.raises(ValueError, match="Duplicate identity"):
        data.boxes(frame, data.annotations[frame["id"]])
    del payload["images"][0]["frame_id"]
    ann.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frame_id"):
        CocoVideo(ann, root, config)


def test_preprocessing_rgb_contract(config):
    pixel = np.zeros((64, 96, 3), np.uint8)
    pixel[:, :, 2] = 255
    image, _ = prepare_image(pixel, config)
    torch.testing.assert_close(
        image[:, 0, 0], torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    )


def test_idf1_switches_and_class_mismatch():
    metric = IdentityMetrics()
    boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2]])
    metric.update([7], boxes, [0], [1], boxes, [0])
    metric.update([7], boxes, [0], [2], boxes, [0])
    result = summarize([metric.counts()])
    assert result["id_switches"] == 1 and result["idf1"] == 0.5
    other = IdentityMetrics()
    other.update([7], boxes, [0], [1], boxes, [1])
    assert other.counts()["tp"] == 0 and other.counts()["idtp"] == 0
    keep = ignore_unmatched(
        boxes,
        boxes,
        torch.tensor([0]),
        torch.tensor([0]),
        [dict(bbox=[0, 0, 100, 100])],
        dict(width=100, height=100),
    )
    assert keep.all()
    keep = ignore_unmatched(
        boxes,
        boxes[:0],
        torch.tensor([]),
        torch.tensor([0]),
        [dict(bbox=[0, 0, 100, 100])],
        dict(width=100, height=100),
    )
    assert not keep.any()


def test_config_fails_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("learning_rte: 1.0")
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(path)


def test_life_labels_keep_occlusion_and_censor_sequence_boundary(config, tmp_path):
    ann, root = make_dataset(tmp_path)
    payload = json.loads(ann.read_text())
    payload["annotations"] = [
        a
        for a in payload["annotations"]
        if not (
            (a["track_id"] == 0 and a["image_id"] > 5)
            or (a["track_id"] == 1 and a["image_id"] in (3, 4, 17, 18))
        )
    ]
    ann.write_text(json.dumps(payload))
    config.clip_length, config.burn_in, config.horizon = 8, 2, 4
    config.clip_step = 2
    config.train_annotations, config.train_images = str(ann), str(root)
    dataset = VideoClips(config)
    first = dataset[0]
    assert not first["present"][2, 1] and first["alive"][2, 1]
    assert first["life_known"][2, 1]  # absent between annotations remains alive
    assert not first["alive"][7, 0] and first["life_known"][7, 0]
    last = dataset[len(dataset) - 1]
    assert not last["alive"][-1, 0]
    assert not last["life_known"][-1, 0]  # only two frames follow its last GT; boundary is censored

import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image


def converter():
    path = Path(__file__).resolve().parents[1] / "tools/convert_visdrone.py"
    spec = importlib.util.spec_from_file_location("visdrone_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset(root):
    sequence = root / "sequences" / "seq01"
    sequence.mkdir(parents=True)
    (root / "annotations").mkdir()
    cv2.imwrite(str(sequence / "000001.jpg"), np.zeros((40, 60, 3), np.uint8))
    cv2.imwrite(str(sequence / "000002.png"), np.zeros((50, 70, 3), np.uint8))
    (root / "annotations" / "seq01.txt").write_text("1,1,5,5,10,10,1,4,0,0\n2,-1,0,0,4,4,0,0,0,0\n")


def test_headers_preserve_dimensions_ignore_and_progress(tmp_path, monkeypatch, capsys):
    dataset(tmp_path)
    module = converter()

    def forbidden(*args):
        raise AssertionError("Fast conversion must not decode pixels")

    monkeypatch.setattr(module.cv2, "imread", forbidden)
    output = tmp_path / "train.json"
    module.convert(tmp_path, output, progress_every=1)
    payload = json.loads(output.read_text())
    assert [(row["height"], row["width"]) for row in payload["images"]] == [
        (40, 60),
        (50, 70),
    ]
    assert payload["annotations"][1]["ignore"] == 1
    assert payload["annotations"][0]["track_id"] == 1
    assert "1/2 frames" in capsys.readouterr().out
    assert not (tmp_path / "train.json.tmp").exists()


def test_decode_and_header_outputs_identical(tmp_path):
    dataset(tmp_path)
    module = converter()
    module.convert(tmp_path, tmp_path / "fast.json")
    module.convert(tmp_path, tmp_path / "verified.json", verify_images=True)
    assert (tmp_path / "fast.json").read_bytes() == (tmp_path / "verified.json").read_bytes()


def test_exif_dimensions_match_loader(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (60, 40))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    module = converter()
    assert module.image_size(path) == module.image_size(path, verify_images=True) == (60, 40)


def test_missing_frame_preserves_existing_output(tmp_path):
    dataset(tmp_path)
    (tmp_path / "annotations" / "seq01.txt").write_text("999,1,5,5,10,10,1,4,0,0\n")
    output = tmp_path / "train.json"
    output.write_text("previous output")
    with pytest.raises(ValueError, match="missing frames"):
        converter().convert(tmp_path, output)
    assert output.read_text() == "previous output"


def test_nested_archive_root_is_resolved_without_walking_images(tmp_path, monkeypatch):
    inner = tmp_path / "VisDrone2019-MOT-train" / "VisDrone2019-MOT-train"
    dataset(inner)
    module = converter()
    original = Path.iterdir

    def guarded(path):
        if path == inner or "sequences" in path.parts or "annotations" in path.parts:
            raise AssertionError("Root detection must stop at a valid root")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", guarded)
    assert module.resolve_root(tmp_path, "train") == inner


def test_split_disambiguates_multiple_roots(tmp_path):
    train = tmp_path / "train" / "raw"
    val = tmp_path / "val" / "raw"
    dataset(train)
    dataset(val)
    module = converter()
    with pytest.raises(ValueError, match="--split"):
        module.resolve_root(tmp_path)
    assert module.resolve_root(tmp_path, "train") == train
    assert module.resolve_root(tmp_path, "val") == val
    module.convert(tmp_path, tmp_path / "val.json", split="val")
    assert len(json.loads((tmp_path / "val.json").read_text())["images"]) == 2


def test_root_search_has_depth_limit_and_missing_path_error(tmp_path):
    inner = tmp_path / "one" / "two" / "train"
    dataset(inner)
    module = converter()
    with pytest.raises(FileNotFoundError, match="within 1 levels"):
        module.resolve_root(tmp_path, max_depth=1)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        module.resolve_root(tmp_path / "missing")


def test_explicit_opposite_split_is_rejected(tmp_path):
    root = tmp_path / "VisDrone2019-MOT-val"
    dataset(root)
    with pytest.raises(FileNotFoundError, match="split="):
        converter().resolve_root(root, "train")

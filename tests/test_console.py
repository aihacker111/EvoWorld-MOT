import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import pytest


def module(name, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(name, tools / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def console(tmp_path, monkeypatch, every=10, mode="pretty"):
    module_ = module("format_train_log", monkeypatch)
    metadata = dict(
        config=dict(
            epochs=40, batch_size=1, grad_accumulation=2, al_min_examples=256, al_critic_min_examples=64
        )
    )
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    output = io.StringIO()
    renderer = module_.TrainingConsole(output, every=every, mode=mode)
    startup = dict(
        run=str(tmp_path),
        parameters=dict(perception=7589870, world=448464, al=76939),
        clips=6016,
        samples_per_epoch=6016,
        world_size=2,
        devices=["cuda:0", "cuda:1"],
        effective_batch=4,
    )
    renderer.consume(json.dumps(startup) + "\n")
    return renderer, output


def record(update, **changes):
    result = dict(
        epoch=1,
        updates=update,
        clips_seen=update * 4,
        seconds=update * 0.1,
        loss=5.302609980106354,
        detector=3.4883607625961304,
        reid=1.3395,
        association=1.16183,
        world=-2.79497,
        grad_norm=10.6474,
        al_policy_total=300,
        al_critic_total=update * 14,
    )
    result.update(changes)
    return result


def test_compact_table_and_raw_metrics_remain_untouched(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch)
    raw = json.dumps(record(1)) + "\n"
    path = tmp_path / "metrics.jsonl"
    path.write_text(raw)
    renderer.consume(raw)
    text = output.getvalue()
    assert "2 GPU" in text and "batch 4" in text and "6016" in text
    assert "5.303" in text and "5.302609980106354" not in text
    assert "World" in text and "ETA ep" in text and "warm" in text
    row = [line for line in text.splitlines() if line.strip().startswith("1/40")][0]
    assert len(row) <= 100
    assert path.read_text() == raw


def test_module_learning_rates_are_compact_and_use_actual_warmup_values(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch)
    renderer.consume(
        json.dumps(
            dict(
                event="learning_rates",
                epoch=1,
                lrs={
                    "detector": 5e-5 / 3,
                    "reid": 3e-4 / 3,
                    "world": 1e-4 / 3,
                    "camera": 1e-4 / 3,
                    "association": 2e-4 / 3,
                    "al": 1e-4 / 3,
                },
            )
        )
        + "\n"
    )
    rows = [line for line in output.getvalue().splitlines() if line.startswith("LR epoch")]
    assert len(rows) == 2 and all(len(row) <= 100 for row in rows)
    assert "detector 1.67e-05" in rows[0] and "reid 1.00e-04" in rows[0]


def test_validation_events_are_readable_and_clear_training_eta(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch)
    renderer.consume(json.dumps(record(1)) + "\n")
    renderer.consume(
        json.dumps(dict(event="validation_start", smoke_test=True, updates=1, max_frames=3)) + "\n"
    )
    renderer.consume(
        json.dumps(dict(event="validation_progress", frames=3, total_frames=3, sequence="test")) + "\n"
    )
    renderer.consume(
        json.dumps(
            dict(
                event="validation_end",
                idf1=0.5,
                mota=0.3,
                id_switches=1,
                frames=3,
                seconds=2,
                truncated=True,
                out=str(tmp_path),
            )
        )
        + "\n"
    )
    text = output.getvalue()
    assert "Validation SMOKE" in text and "3/3 frames" in text
    assert "Validation OK | IDF1 50.00%" in text and "partial" in text
    assert renderer.last is None and renderer.rate is None


def test_throttles_rows_but_always_flushes_last_update(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch, every=10)
    for i in range(1, 26):
        renderer.consume(json.dumps(record(i)) + "\n")
    renderer.finish(0)
    rows = [line for line in output.getvalue().splitlines() if line.strip().startswith("1/40")]
    assert [int(line.split()[1]) for line in rows] == [1, 2, 3, 10, 20, 25]


def test_resume_and_epoch_boundary_do_not_inflate_rate(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch, every=1)
    renderer.consume(json.dumps(record(1001, epoch=2, clips_seen=4000, seconds=1)) + "\n")
    assert renderer.rate is None
    renderer.consume(json.dumps(record(1002, epoch=2, clips_seen=4004, seconds=3)) + "\n")
    assert renderer.rate == 2.0
    assert "16m46s" in output.getvalue()
    renderer.consume(json.dumps(record(1505, epoch=3, clips_seen=4, seconds=4000)) + "\n")
    assert renderer.rate is None


def test_amp_event_al_readiness_and_unknown_output(tmp_path, monkeypatch):
    renderer, output = console(tmp_path, monkeypatch, every=1)
    renderer.consume(json.dumps(dict(event="amp_overflow", epoch=1, updates=0, scale=32768)) + "\n")
    renderer.consume(json.dumps(record(1)) + "\n")
    renderer.consume(json.dumps(record(5)) + "\n")
    renderer.consume("third-party diagnostic\n")
    text = output.getvalue()
    assert "AMP skip" in text and "32,768" in text
    assert "warm" in text and "ready" in text and "third-party diagnostic" in text


def test_json_passthrough_is_byte_identical(monkeypatch):
    module_ = module("format_train_log", monkeypatch)
    output = io.StringIO()
    renderer = module_.TrainingConsole(output, mode="json")
    raw = json.dumps(record(1)) + "\n"
    renderer.consume(raw)
    renderer.consume("diagnostic\n")
    renderer.finish(0)
    assert output.getvalue() == raw + "diagnostic\n"


def test_child_exit_code_and_stderr_are_preserved(tmp_path, monkeypatch, capfd):
    launcher = module("run_training", monkeypatch)
    renderer, output = console(tmp_path, monkeypatch)
    command = [
        sys.executable,
        "-u",
        "-c",
        'import sys; print("diagnostic"); print("failure detail", file=sys.stderr); sys.exit(7)',
    ]
    assert launcher.run_command(command, renderer) == 7
    assert "diagnostic" in output.getvalue() and "exit code 7" in output.getvalue()
    assert "failure detail" in capfd.readouterr().err


def test_interrupt_terminates_owned_child_group(tmp_path, monkeypatch):
    launcher = module("run_training", monkeypatch)

    class Console:
        code = None

        def consume(self, line):
            raise KeyboardInterrupt

        def finish(self, code):
            self.code = code

    renderer = Console()
    started = time.monotonic()
    result = launcher.run_command(
        [sys.executable, "-u", "-c", 'import time; print("ready", flush=True); time.sleep(30)'], renderer
    )
    assert result == 130 and renderer.code == 130
    assert time.monotonic() - started < 5


def test_invalid_log_interval_rejected(monkeypatch):
    with pytest.raises(ValueError, match="positive"):
        module("format_train_log", monkeypatch).TrainingConsole(every=0)


def test_pretty_launcher_trains_and_resumes_without_source_changes(tmp_path):
    import os
    import subprocess

    pytest.importorskip("evoworld")
    import yaml
    from evoworld.config import Config, fingerprint
    from evoworld.synthetic import make_dataset

    annotation, images = make_dataset(tmp_path / "data")
    config = Config(
        backend="tiny",
        width=96,
        height=64,
        hidden_dim=32,
        embedding_dim=16,
        stochastic_dim=4,
        camera_dim=16,
        clip_length=6,
        burn_in=1,
        horizon=2,
        probe_horizon=2,
        workers=0,
        epochs=3,
        grad_accumulation=1,
        clip_step=2,
        train_annotations=str(annotation),
        train_images=str(images),
        out=str(tmp_path / "run"),
    )
    before = fingerprint(config)
    cfg = tmp_path / "train.yaml"
    cfg.write_text(yaml.safe_dump(config.dictionary()))
    launcher = Path(__file__).resolve().parents[1] / "tools/run_training.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(launcher), "--config", str(cfg), "--max-steps", "2"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "EvoWorld-MOT | TRAIN" in completed.stdout and "1/3" in completed.stdout
    assert '"detector":' not in completed.stdout
    log = Path(config.out) / "metrics.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    updates = [row for row in records if "loss" in row]
    assert len(updates) == 2 and updates[-1]["updates"] == 2 and "al_memory" in updates[-1]
    checkpoint = Path(config.out) / "last.pt"
    resumed = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--config",
            str(cfg),
            "--max-steps",
            "3",
            "--resume",
            str(checkpoint),
            "--log-format",
            "quiet",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    assert not resumed.stdout.strip()
    assert json.loads(log.read_text().splitlines()[-1])["updates"] == 3
    assert fingerprint(config) == before

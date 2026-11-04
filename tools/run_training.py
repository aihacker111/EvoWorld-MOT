#!/usr/bin/env python3
"""Launch unchanged evoworld training with a compact, checkpoint-independent console."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from format_train_log import TrainingConsole


def command_line(training_args, distributed=False, port=29517):
    if distributed:
        return [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "-m",
            "evoworld",
            "train",
            *training_args,
        ]
    return [sys.executable, "-u", "-m", "evoworld", "train", *training_args]


def stop_child(process):
    if process.poll() is not None:
        return
    # The launcher owns this fresh process group. torchrun forwards termination to workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=15)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_command(command, console, cwd=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    ) as process:
        try:
            for line in process.stdout:
                console.consume(line)
            returncode = process.wait()
        except KeyboardInterrupt:
            stop_child(process)
            console.finish(130)
            return 130
        except BaseException:
            stop_child(process)
            raise
    console.finish(returncode)
    return returncode if returncode >= 0 else 128 - returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--distributed", action="store_true", help="Launch two torchrun ranks")
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-format", choices=("pretty", "json", "quiet"), default="pretty")
    args, training_args = parser.parse_known_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    console = TrainingConsole(every=args.log_every, mode=args.log_format)
    command = command_line(training_args, args.distributed, args.master_port)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return run_command(command, console, Path(__file__).resolve().parents[1])
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())

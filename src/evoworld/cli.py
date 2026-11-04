"""Command-line entry points. Run python -m evoworld --help."""

import argparse
import json
from evoworld.config import Config, resolve


def main():
    parser = argparse.ArgumentParser(prog="evoworld")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Jointly train the complete model")
    training.add_argument("--config", default="configs/full.yaml")
    training.add_argument("--resume")
    training.add_argument("--init-checkpoint", help="Transfer v1/v2 weights into a fresh AL run")
    training.add_argument("--out")
    training.add_argument("--batch-size", type=int, help="Clips per GPU/process")
    training.add_argument("--grad-accumulation", type=int)
    training.add_argument("--workers", type=int, help="DataLoader workers per GPU/process")
    training.add_argument("--max-steps", type=int, help="Stop after this total number of optimizer updates")
    training.add_argument(
        "--validate-after-steps",
        type=int,
        help="Run this many additional successful optimizer updates, validate, then exit (smoke test)",
    )
    training.add_argument(
        "--val-max-frames",
        type=int,
        help="Limit total validation frames in --validate-after-steps mode; omit for full validation",
    )
    inspect = commands.add_parser("inspect", help="Strictly load detector and report parameter counts")
    inspect.add_argument("--config", default="configs/full.yaml")
    synthetic = commands.add_parser("synthetic", help="Generate a tiny deterministic test dataset")
    synthetic.add_argument("--out", default="data/synthetic")
    smoke = commands.add_parser("smoke", help="Exercise training on synthetic data; no quality claim")
    smoke.add_argument("--out", default="runs/smoke")
    smoke.add_argument("--backend", choices=["tiny", "falcon"], default="tiny")
    smoke.add_argument("--steps", type=int, default=2)
    for name in ("track", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--out", default=f"runs/{name}")
        command.add_argument("--device", default="cpu")
        if name == "track":
            command.add_argument("--source", required=True)
            command.add_argument("--fps", type=float, default=25)
        else:
            command.add_argument("--annotations", required=True)
            command.add_argument("--images", required=True)
            command.add_argument("--hota", action="store_true")
    args = parser.parse_args()
    if args.command == "synthetic":
        from evoworld.synthetic import make_dataset

        print(make_dataset(resolve(args.out)))
    elif args.command == "inspect":
        from evoworld.runtime import setup
        from evoworld.models.system import EvoWorld

        config = Config.load(resolve(args.config))
        setup(config)
        model = EvoWorld(config)
        print(
            json.dumps(
                {"detector": model.perception.load_report, "parameters": model.parameter_report()}, indent=2
            )
        )
    elif args.command in ("train", "smoke"):
        from evoworld.train import train

        if args.command == "smoke":
            from evoworld.synthetic import make_dataset

            annotation, images = make_dataset(resolve(args.out) / "synthetic")
            config = Config(
                backend=args.backend,
                width=96,
                height=64,
                hidden_dim=32,
                stochastic_dim=4,
                embedding_dim=16,
                camera_dim=16,
                clip_length=6,
                burn_in=1,
                horizon=2,
                probe_horizon=2,
                clip_step=2,
                workers=0,
                epochs=3,
                batch_size=1,
                grad_accumulation=1,
                probe_every=1,
                effect_min_examples=12,
                train_annotations=str(annotation),
                train_images=str(images),
                out=args.out,
                learning_rate=0.001,
                perception_lr=0.00001,
            )
            checkpoint = train(config.validate(), max_steps=args.steps)
        else:
            config = Config.load(resolve(args.config))
            if args.out:
                config.out = args.out
            if args.max_steps is not None and args.max_steps < 1:
                parser.error("--max-steps must be positive")
            if args.validate_after_steps is not None and args.validate_after_steps < 1:
                parser.error("--validate-after-steps must be positive")
            if args.val_max_frames is not None and args.val_max_frames < 1:
                parser.error("--val-max-frames must be positive")
            if args.val_max_frames is not None and args.validate_after_steps is None:
                parser.error("--val-max-frames requires --validate-after-steps")
            if args.validate_after_steps is not None and args.max_steps is not None:
                parser.error("Choose --validate-after-steps or --max-steps")
            for option in ("batch_size", "grad_accumulation", "workers"):
                value = getattr(args, option)
                if value is not None:
                    setattr(config, option, value)
            config.validate()
            checkpoint = train(
                config,
                args.resume,
                args.max_steps,
                args.init_checkpoint,
                validate_after_steps=args.validate_after_steps,
                val_max_frames=args.val_max_frames,
            )
        import os

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Checkpoint: {checkpoint}")
    else:
        from evoworld.runtime import load_for_inference

        model, _ = load_for_inference(resolve(args.checkpoint), args.device)
        if args.command == "track":
            from evoworld.inference import track

            result = track(model, args.source, resolve(args.out), args.device, args.fps)
        else:
            from evoworld.evaluate import evaluate

            result = evaluate(model, args.annotations, args.images, resolve(args.out), args.device, args.hota)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

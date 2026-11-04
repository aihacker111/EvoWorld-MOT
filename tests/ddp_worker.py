"""Real two-process Gloo integration harness, launched with torchrun by tests."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from evoworld.config import Config
from evoworld.distributed import DistributedContext
from evoworld.train import train


def verify_conditional_accumulation():
    """Optional replay heads may be used only in an earlier no_sync microbatch."""
    from torch.nn.parallel import DistributedDataParallel

    class Objective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0))
            self.optional = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, first):
            loss = self.base * 1.0
            if first and dist.get_rank() == 0:
                loss = loss + self.optional
            return loss

    model = Objective()
    parallel = DistributedDataParallel(model, find_unused_parameters=True)
    with parallel.no_sync():
        (parallel(True) / 2).backward()
    (parallel(False) / 2).backward()
    torch.testing.assert_close(model.base.grad, torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(model.optional.grad, torch.tensor(0.25), rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume")
    args = parser.parse_args()
    dist.init_process_group("gloo")
    verify_conditional_accumulation()
    captured = []
    original = DistributedContext.wrap

    def capture(self, model, objective):
        captured.append(model)
        return original(self, model, objective)

    DistributedContext.wrap = capture
    try:
        path = train(Config.load(args.config), resume=args.resume, max_steps=args.steps)
        model = captured[0]
        vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        vectors = [torch.empty_like(vector) for _ in range(dist.get_world_size())]
        dist.all_gather(vectors, vector)
        assert all(torch.equal(vector, other) for other in vectors), "DDP parameters diverged across ranks"
        counts = torch.tensor(
            [
                int(model.association.effect_examples),
                int(model.al.policy_examples),
                int(model.al.critic_examples),
            ]
        )
        all_counts = [torch.empty_like(counts) for _ in range(dist.get_world_size())]
        dist.all_gather(all_counts, counts)
        assert all(torch.equal(counts, other) for other in all_counts), "Global AL counters diverged"
        if dist.get_rank() == 0:
            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["distributed"]["world_size"] == 2
            assert len(checkpoint["distributed"]["rng_by_rank"]) == 2
            Path(path).with_suffix(".ddp.json").write_text(
                json.dumps(
                    dict(
                        world_size=2,
                        synchronized=True,
                        parameter_hash=hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
                        counters=counts.tolist(),
                    )
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

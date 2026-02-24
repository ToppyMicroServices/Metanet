"""run_demo.py – end-to-end demonstration of the Safe MetaNet loop.

Usage
-----
From the repository root::

    python scripts/run_demo.py
    python scripts/run_demo.py --config configs/default.yaml --steps 20
    python scripts/run_demo.py --disable-rollback
    python scripts/run_demo.py --disable-scope-expansion
    python scripts/run_demo.py --adapter linear_adapter

No dataset download is required.  Inputs are synthetic random tensors.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from safe_metanet import (
    SafeMetaNetConfig,
    SafeMetaNet,
    build_backbone,
    load_config,
)


def make_dummy_stream(
    num_samples: int,
    batch_size: int,
    input_dim: int,
    seed: int = 42,
):
    """Generate a stream of random input tensors (no labels)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    for _ in range(num_samples):
        yield torch.randn(batch_size, input_dim, generator=rng)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe MetaNet demo")
    p.add_argument("--config", default=None, help="Path to a YAML config file")
    p.add_argument("--steps", type=int, default=15, help="Number of test steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--adapter",
        choices=["lora", "linear_adapter"],
        default=None,
        help="Override adapter type",
    )
    p.add_argument(
        "--disable-rollback",
        action="store_true",
        help="Ablation: disable rollback (always accept updates)",
    )
    p.add_argument(
        "--disable-scope-expansion",
        action="store_true",
        help="Ablation: never expand editable scope",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config (YAML or default dataclass).
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SafeMetaNetConfig()

    # Apply CLI overrides.
    if args.adapter:
        cfg.adapter_type = args.adapter
    if args.disable_rollback:
        cfg.disable_rollback = True
    if args.disable_scope_expansion:
        cfg.disable_scope_expansion = True

    print("=" * 60)
    print("Safe MetaNet – test-time weight editing demo")
    print("=" * 60)
    print(f"  backbone           : {cfg.backbone}")
    print(f"  adapter_type       : {cfg.adapter_type}")
    print(f"  lora_rank          : {cfg.lora_rank}")
    print(f"  disable_rollback   : {cfg.disable_rollback}")
    print(f"  disable_expansion  : {cfg.disable_scope_expansion}")
    print(f"  expansion_schedule : {cfg.expansion_schedule}")
    print(f"  test steps         : {args.steps}")
    print("=" * 60)

    # Build frozen backbone + Safe MetaNet runner.
    torch.manual_seed(args.seed)
    model = build_backbone(cfg)
    runner = SafeMetaNet(model, cfg)

    # Simulate sequential test inputs (no labels).
    data_stream = make_dummy_stream(
        num_samples=args.steps,
        batch_size=args.batch_size,
        input_dim=cfg.input_dim,
        seed=args.seed,
    )

    print("\n[Running Safe MetaNet loop …]\n")
    for idx, x in enumerate(data_stream):
        result = runner.step(x, sample_idx=idx)
        # Individual step logs are emitted by the logger; we only print a
        # compact line here if the log level is above DEBUG.
        action_sym = "✓" if result["action"] == "accept" else "✗"
        print(
            f"  [{action_sym}] step {idx+1:3d}  "
            f"metric {result['metric_before']:.4f} → {result['metric_after']:.4f}"
            + (" [expanded]" if result["scope_expanded"] else "")
        )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(runner.summary())


if __name__ == "__main__":
    main()

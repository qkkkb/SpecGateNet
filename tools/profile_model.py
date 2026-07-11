#!/usr/bin/env python3
"""Profile the checkpoint-compatible SpecGateNet model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SpecGateNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    model = SpecGateNet(
        size=args.size,
        n_classes=args.num_classes,
        cnn_pretrained=False,
        aux_loss=False,
    ).to(device).eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("model=SpecGateNet")
    print(f"num_classes={args.num_classes}")
    print(f"input_shape=(1, 3, {args.size}, {args.size})")
    print(f"parameters={total_params} ({total_params / 1e6:.6f} M)")
    print(f"trainable_parameters={trainable_params} ({trainable_params / 1e6:.6f} M)")

    try:
        from thop import profile
    except ImportError:
        print("FLOPs=not computed (install thop to enable FLOP profiling)")
        return

    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    with torch.inference_mode():
        flops, thop_params = profile(model, inputs=(dummy,), verbose=False)
    print(f"thop_parameters={thop_params:.0f} ({thop_params / 1e6:.6f} M)")
    print(f"thop_flops={flops:.0f} ({flops / 1e9:.6f} G)")


if __name__ == "__main__":
    main()

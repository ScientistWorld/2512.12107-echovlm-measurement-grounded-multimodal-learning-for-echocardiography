"""Simple CLIP-only baseline using the same architecture without EchoVLM losses."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/echo_toy")
    parser.add_argument("--output-dir", default="checkpoints/clip_only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "method/train.py",
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lambda-view",
        "0.0",
        "--lambda-neg",
        "0.0",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

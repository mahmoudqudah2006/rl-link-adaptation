from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train_q_learning


def main() -> None:
    parser = argparse.ArgumentParser(description="RL link adaptation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output", type=Path, default=Path("results/q_learning.json"))
    args = parser.parse_args()

    _, result = train_q_learning(args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

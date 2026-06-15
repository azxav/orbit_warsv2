from __future__ import annotations

import argparse
import json

from .evaluate import evaluate
from .export_agent import export_agent
from .train_loop import train


def _cmd_train(args: argparse.Namespace) -> None:
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


def _cmd_eval(args: argparse.Namespace) -> None:
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


def _cmd_export(args: argparse.Namespace) -> None:
    export_agent(args.checkpoint, args.out)
    print(json.dumps({"out": args.out}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m orbit_board_bc_train.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--hidden-dim", type=int, default=192)
    tr.add_argument("--encoder-layers", type=int, default=4)
    tr.add_argument("--decoder-layers", type=int, default=2)
    tr.add_argument("--heads", type=int, default=6)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--noop-stop-weight", type=float, default=0.35)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--num-workers", type=int, default=0)
    tr.add_argument("--pin-memory", action="store_true", default=False)
    tr.add_argument("--persistent-workers", action="store_true", default=False)
    tr.add_argument("--prefetch-factor", type=int, default=2)
    tr.add_argument("--shuffle-block-size", type=int, default=65536)
    tr.add_argument("--log-file", help="Training log path. Defaults to <out-dir>/train.log")
    tr.add_argument("--log-interval", type=int, default=10, help="Write batch progress every N batches")
    tr.add_argument("--resume", help="Training checkpoint to resume from, usually <out-dir>/last.pt")
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--batch-size", type=int, default=128)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--num-workers", type=int, default=0)
    ev.add_argument("--pin-memory", action="store_true", default=False)
    ev.add_argument("--persistent-workers", action="store_true", default=False)
    ev.add_argument("--prefetch-factor", type=int, default=2)
    ev.set_defaults(func=_cmd_eval)

    ex = sub.add_parser("export-agent")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--out", required=True)
    ex.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

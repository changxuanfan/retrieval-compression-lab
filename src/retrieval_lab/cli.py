"""Command line entry points. Invoke from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="COMP5331 first-four compression baselines")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "embed", "run", "pipeline"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name in {"run", "pipeline"}:
            command.add_argument("--split", choices=["train", "dev", "test"])
    demo_parser = commands.add_parser("demo", help="Synthetic offline pipeline check")
    demo_parser.add_argument("--output", default="runs/demo")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--study", required=True)
    query_parser = commands.add_parser(
        "query", help="Search new text with a saved compressor/index"
    )
    query_parser.add_argument("--run", required=True, help="Run directory containing index.faiss")
    query_parser.add_argument("--text", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "pipeline":
        # Keep ST/PyTorch model encoding separate from native FAISS evaluation.
        for stage in ["prepare", "embed", "run"]:
            command = [sys.executable, "-m", "retrieval_lab.cli", stage, "--config", args.config]
            if stage == "run" and args.split:
                command.extend(["--split", args.split])
            subprocess.run(command, check=True)
        return
    if args.command == "query":
        from .query import query

        query(Path(args.run), args.text, args.top_k)
        return
    from .reporting import write_report

    if args.command == "demo":
        from .experiment import demo

        print(write_report(demo(args.output)))
        return
    if args.command == "report":
        print(write_report(Path(args.study)))
        return
    config = load_config(args.config)
    if getattr(args, "split", None):
        config["experiment"]["split"] = args.split
    from .data import load_dataset, prepare

    if args.command in {"prepare", "pipeline"}:
        prepare(config)
    if args.command == "prepare":
        return
    folder = Path(config["data"]["directory"]) / config["data"]["name"]
    dataset = load_dataset(folder, config["data"])
    from .embedding import cache_path, embed

    if args.command in {"embed", "pipeline"}:
        embed(config, dataset)
    if args.command == "embed":
        return
    from .experiment import run

    study = run(config, dataset, cache_path(config, dataset))
    print(write_report(study))


if __name__ == "__main__":
    main()

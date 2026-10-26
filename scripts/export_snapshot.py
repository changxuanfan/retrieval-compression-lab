"""Export small measured artifacts for Git/report use, excluding models/data/indexes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from retrieval_lab.common import read_json, sha256, write_json
from retrieval_lab.metrics import paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in [
        "report.md",
        "summary.csv",
        "summary.json",
        "runs.csv",
        "quality-memory.png",
        "case-selection.json",
        "study.json",
    ]:
        shutil.copy2(args.study / name, args.output / name)
    study = read_json(args.study / "study.json")
    shutil.copy2(
        Path(study["embedding_cache"]) / "manifest.json", args.output / "embedding-manifest.json"
    )
    results, per_query = {}, {}
    for path in sorted(args.study.glob("*/result.json")):
        results[path.parent.name] = read_json(path)
        per_query[path.parent.name] = read_json(path.parent / "per_query.json")
    write_json(args.output / "run-results.json", results)
    write_json(args.output / "per-query.json", per_query)
    comparisons = []
    for name, result in results.items():
        comparator = f"pca-{result['dimension']}d-seed0"
        if result["method"] == "ae" and comparator in results:
            comparisons.append(
                {
                    "candidate": name,
                    "reference": comparator,
                    **paired_bootstrap(per_query[comparator], per_query[name], "ndcg@10"),
                }
            )
    write_json(args.output / "paired-comparisons.json", comparisons)
    write_json(
        args.output / "snapshot.json",
        {
            "source_study": str(args.study),
            "source_sha256": study["source_sha256"],
            "scope": f"{next(iter(results.values()))['dataset']} / "
            f"{study['config']['experiment']['split']}; "
            "see original config for dimension/seed/training budget",
            "files": {
                path.name: sha256(path)
                for path in args.output.iterdir()
                if path.is_file() and path.name != "snapshot.json"
            },
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

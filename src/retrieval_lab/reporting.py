"""Generate report-ready tables/figures from measured runs, never placeholder scores."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, write_json


def write_report(root: Path) -> Path:
    study = read_json(root / "study.json")
    results = [(path.parent, read_json(path)) for path in sorted(root.glob("*/result.json"))]
    if not results:
        raise ValueError(f"No completed results in {root}")
    groups = defaultdict(list)
    records = []
    for folder, result in results:
        groups[(result["method"], result["dimension"])].append(result)
        records.append(
            {
                "method": result["method"],
                "dimension": result["dimension"],
                "seed": result["seed"],
                **result["metrics"],
                "query_doc_overlap@10": result["query_doc_overlap@10"],
                "doc_doc_overlap@10": result["doc_doc_overlap@10"],
                **result["storage"],
                **{
                    key: value
                    for key, value in result["timing"].items()
                    if isinstance(value, (int, float))
                },
                "run_directory": folder.name,
            }
        )
    with (root / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    for (method, dim), runs in sorted(groups.items()):
        summary = {"method": method, "dimension": dim, "n_seeds": len(runs)}
        metrics = list(runs[0]["metrics"]) + ["query_doc_overlap@10", "doc_doc_overlap@10"]
        for metric in metrics:
            values = [run["metrics"].get(metric, run.get(metric)) for run in runs]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary["vector_payload_bytes"] = runs[0]["storage"]["vector_payload_bytes"]
        summary["index_file_bytes_mean"] = float(
            np.mean([run["storage"]["index_file_bytes"] for run in runs])
        )
        summary["search_p50_ms_mean"] = float(
            np.mean([run["timing"]["search_p50_ms"] for run in runs])
        )
        summaries.append(summary)
    write_json(root / "summary.json", summaries)
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    metric = (
        "ndcg@10" if "ndcg@10" in results[0][1]["metrics"] else next(iter(results[0][1]["metrics"]))
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    originals = [row for row in summaries if row["method"] == "original"]
    if originals:
        axes[0].axhline(
            originals[0][f"{metric}_mean"], color="gray", linestyle="--", label="original"
        )
    for method in ["pca", "rp", "ae"]:
        rows = sorted(
            (row for row in summaries if row["method"] == method), key=lambda row: row["dimension"]
        )
        if not rows:
            continue
        axes[0].errorbar(
            [r["dimension"] for r in rows],
            [r[f"{metric}_mean"] for r in rows],
            yerr=[r[f"{metric}_std"] for r in rows],
            marker="o",
            label=method,
        )
        axes[1].plot(
            [r["vector_payload_bytes"] / 2**20 for r in rows],
            [r[f"{metric}_mean"] for r in rows],
            marker="o",
            label=method,
        )
    axes[0].set(xlabel="Embedding dimension", ylabel=metric)
    axes[1].set(xlabel="Float32 corpus vector payload (MiB)", ylabel=metric)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"{results[0][1]['dataset']} / {results[0][1]['split']} / {study['study_id']}")
    fig.savefig(root / "quality-memory.png", dpi=180)
    plt.close(fig)

    # IDs and scores expose both improvements and regressions without cherry-picking.
    original_folder = next(
        (folder for folder, result in results if result["method"] == "original"), None
    )
    cases = []
    if original_folder:
        reference = read_json(original_folder / "per_query.json")
        for folder, result in results:
            if result["method"] == "original":
                continue
            candidate = read_json(folder / "per_query.json")
            differences = sorted(
                (candidate[qid][metric] - reference[qid][metric], qid) for qid in reference
            )
            cases.append(
                {
                    "run": folder.name,
                    "metric": metric,
                    "largest_regressions": [
                        {"query_id": qid, "delta": delta} for delta, qid in differences[:3]
                    ],
                    "largest_improvements": [
                        {"query_id": qid, "delta": delta} for delta, qid in differences[-3:][::-1]
                    ],
                }
            )
    write_json(root / "case-selection.json", cases)
    synthetic = results[0][1]["dataset"] == "SYNTHETIC"
    text = [
        "# Measured experiment summary",
        "",
        "SYNTHETIC PIPELINE CHECK ONLY. Not a BEIR result."
        if synthetic
        else "Measured on the complete official corpus; no relevance labels used for fitting.",
        "",
        f"Study ID: `{study['study_id']}`; split: `{results[0][1]['split']}`.",
        "",
        f"| Method | Dimension | Seeds | {metric} mean ± sample std | Search p50 mean (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        text.append(
            f"| {row['method']} | {row['dimension']} | {row['n_seeds']} | "
            f"{row[f'{metric}_mean']:.5f} ± {row[f'{metric}_std']:.5f} | "
            f"{row['search_p50_ms_mean']:.3f} |"
        )
    text += [
        "",
        "![Quality and memory](quality-memory.png)",
        "",
        "Original/PCA are deterministic fits and run once; std=0 is not a confidence bound.",
        "AE/RP std describes seed variation. Per-run query bootstrap intervals are in result.json.",
        "Search latency excludes Transformer encoding and compression; no end-to-end speedup claim.",
        "Inspect runs.csv, summary.csv, per_query.json, ranking.jsonl and case-selection.json.",
        "Training history is reconstruction on a corpus-only diagnostic probe, not validation retrieval.",
        "Model/data/source hashes and full configuration are in study.json and the embedding manifest.",
    ]
    (root / "report.md").write_text("\n".join(text) + "\n")
    return root / "report.md"

"""Build scientific diagnostics and balanced failure examples from saved dev runs."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

from retrieval_lab.common import read_json, sha256, source_digest, write_json
from retrieval_lab.data import load_dataset
from retrieval_lab.metrics import paired_bootstrap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = read_json(args.audit / "diagnostics.json")
    summary = read_json(args.study / "summary.json")
    study = read_json(args.study / "study.json")
    settings = study["config"]["data"]
    dataset = load_dataset(Path(settings["directory"]) / settings["name"], settings)
    per_query = {path.parent.name: read_json(path) for path in args.study.glob("*/per_query.json")}
    dimensions = sorted({row["dimension"] for row in diagnostics["ae"]})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    gaps = [
        [
            100 * (r["reconstruction_ratio_to_pca"] - 1)
            for r in diagnostics["ae"]
            if r["dimension"] == dim
        ]
        for dim in dimensions
    ]
    axes[0, 0].bar(
        [str(dim) for dim in dimensions],
        [np.mean(v) for v in gaps],
        yerr=[np.std(v, ddof=1) for v in gaps],
        capsize=4,
    )
    axes[0, 0].set(
        xlabel="AE latent dimension",
        ylabel="Reconstruction excess over PCA (%)",
        title="Full-corpus reconstruction; seed mean / SD",
    )
    paired = diagnostics["ae_vs_pca_seed_query_bootstrap"]
    means = [r["delta"] for r in paired]
    intervals = np.array([r["seed_query_bootstrap_95ci"] for r in paired])
    axes[0, 1].errorbar(
        dimensions,
        means,
        yerr=[np.array(means) - intervals[:, 0], intervals[:, 1] - means],
        fmt="o",
        capsize=4,
    )
    axes[0, 1].axhline(0, color="gray", linestyle="--")
    axes[0, 1].set(
        xlabel="Dimension",
        ylabel="AE minus PCA nDCG@10",
        title="Exploratory seed + query bootstrap 95% interval",
    )
    for method, marker in [("pca", "s"), ("ae", "o"), ("rp", "^")]:
        rows = [row for row in summary if row["method"] == method and row["dimension"] == 128]
        axes[1, 0].scatter(
            [r["query_doc_overlap@10_mean"] for r in rows],
            [r["ndcg@10_mean"] for r in rows],
            marker=marker,
            label=method,
            s=65,
        )
    axes[1, 0].set(
        xlabel="Query-document overlap@10",
        ylabel="nDCG@10",
        title="Same 128d: geometry vs relevance (seed means)",
    )
    axes[1, 0].legend()
    for method, field, marker in [
        ("raw AE", "raw_ae_ndcg@10", "o"),
        ("orthonormalized AE (diagnostic)", "orthonormalized_rowspace_ndcg@10", "s"),
        ("PCA", "pca_ndcg@10", "^"),
    ]:
        values = [
            np.mean([r[field] for r in diagnostics["ae"] if r["dimension"] == dim])
            for dim in dimensions
        ]
        axes[1, 1].plot(dimensions, values, marker=marker, label=method)
    axes[1, 1].set(
        xlabel="Dimension",
        ylabel="nDCG@10",
        title="Changing latent coordinates while retaining AE rowspace",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("SciFact dev audit; 162 queries; 3 seeds; diagnostics are not new baselines")
    fig.savefig(args.audit / "diagnostics.png", dpi=180)
    plt.close(fig)

    pca = per_query["pca-128d-seed0"]
    ae_names = [name for name in per_query if name.startswith("ae-128d-")]
    differences = sorted(
        (np.mean([per_query[name][qid]["ndcg@10"] for name in ae_names]) - pca[qid]["ndcg@10"], qid)
        for qid in pca
    )
    rankings = {}
    for name in ["original-768d-seed0", "pca-128d-seed0", *ae_names]:
        rankings[name] = {
            row["query_id"]: row["document_ids"]
            for row in (
                json.loads(line)
                for line in (args.study / name / "ranking.jsonl").read_text().splitlines()
            )
        }
    cases = []
    for category, chosen in [
        ("regression", differences[:3]),
        ("improvement", differences[-3:][::-1]),
    ]:
        for delta, qid in chosen:
            relevant = [did for did, grade in dataset.qrels["dev"][qid].items() if grade > 0]
            cases.append(
                {
                    "category": category,
                    "query_id": qid,
                    "query_text": dataset.queries[qid],
                    "ae_mean_minus_pca_ndcg@10": float(delta),
                    "relevant_documents": [
                        {"id": did, "title": dataset.corpus[did].get("title", "")}
                        for did in relevant
                    ],
                    "runs": {
                        name: {
                            "ndcg@10": per_query[name][qid]["ndcg@10"],
                            "relevant_ranks_in_top100": {
                                did: (docs[qid].index(did) + 1 if did in docs[qid] else None)
                                for did in relevant
                            },
                            "top10": docs[qid][:10],
                        }
                        for name, docs in rankings.items()
                    },
                    "interpretation": "Selected by a symmetric numeric rule, not causal explanation",
                }
            )
    write_json(args.audit / "balanced-cases.json", cases)
    write_json(
        args.audit / "pca-tradeoff-comparisons.json",
        {
            "pca256_minus_pca384": paired_bootstrap(
                per_query["pca-384d-seed0"], per_query["pca-256d-seed0"], "ndcg@10", 10000
            ),
            "pca128_minus_pca256": paired_bootstrap(
                per_query["pca-256d-seed0"], per_query["pca-128d-seed0"], "ndcg@10", 10000
            ),
            "pca256_minus_original": paired_bootstrap(
                per_query["original-768d-seed0"], per_query["pca-256d-seed0"], "ndcg@10", 10000
            ),
        },
    )
    extra = {}
    for file in sorted(args.study.glob("ae-*/*")):
        if file.name in {"training_decoder.npz", "training.csv"}:
            extra[str(file.relative_to(args.study))] = sha256(file)
    capsule = {
        "training_auxiliary_file_sha256_at_audit_time": extra,
        "note": "These new hashes protect subsequent verification; they cannot retroactively "
        "prove auxiliary files were unchanged since original training.",
        "original_core_source_sha256": study["source_sha256"],
        "current_core_source_matches_original": source_digest() == study["source_sha256"],
    }
    if capsule["current_core_source_matches_original"]:
        repository = Path(__file__).resolve().parents[1]
        archive = args.audit / "original-source.zip"
        files = list((repository / "src/retrieval_lab").glob("*.py"))
        files += list((repository / "configs").glob("*.toml"))
        files += [repository / name for name in ["pyproject.toml", "uv.lock", ".python-version"]]
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for file in sorted(files):
                output.write(file, file.relative_to(repository))
            output.write(args.study / "study.json", "measured-study.json")
        capsule["source_archive_sha256"] = sha256(archive)
    write_json(args.audit / "supplementary-provenance.json", capsule)
    print(args.audit)


if __name__ == "__main__":
    main()

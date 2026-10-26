"""Independently replay saved results and diagnose AE/PCA geometry on dev only.

This audit does not alter any fitted model or original result. All additional
variants are labeled diagnostics, not new baselines or test-set evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from retrieval_lab.common import fingerprint, read_json, sha256, unit_vectors, write_json
from retrieval_lab.compressors import Compressor
from retrieval_lab.data import load_dataset
from retrieval_lab.embedding import validate_cache
from retrieval_lab.metrics import evaluate, overlap


def independent_metrics(qrels, rankings, ks):
    """Scalar Python reference, independent from retrieval_lab.metrics arrays."""
    import math

    scores = {}
    for qid, labels in qrels.items():
        row = {}
        positives = {did for did, grade in labels.items() if grade > 0}
        for k in ks:
            top = rankings.get(qid, [])[:k]
            ideal = sorted([v for v in labels.values() if v > 0], reverse=True)[:k]
            dcg = sum(labels.get(did, 0) / math.log2(rank + 2) for rank, did in enumerate(top))
            idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
            row[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
            row[f"recall@{k}"] = len(set(top) & positives) / len(positives) if positives else 0.0
            row[f"mrr@{k}"] = next(
                (1 / (rank + 1) for rank, did in enumerate(top) if did in positives), 0.0
            )
        scores[qid] = row
    return scores


def seed_query_bootstrap(differences, samples=10000, seed=2026):
    """Independently resample seeds and shared queries; 3 seeds is still limited."""
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences, dtype=float)
    count_seeds, count_queries = differences.shape
    means = np.empty(samples)
    for offset in range(0, samples, 128):
        count = min(128, samples - offset)
        seeds = rng.integers(count_seeds, size=(count, count_seeds, 1))
        queries = rng.integers(count_queries, size=(count, 1, count_queries))
        means[offset : offset + count] = differences[seeds, queries].mean(axis=(1, 2))
    return {
        "delta": float(differences.mean()),
        "seed_query_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "seed_count": count_seeds,
        "query_count": count_queries,
        "samples": samples,
        "bootstrap_seed": seed,
        "limitation": "Exploratory percentile interval; only three trained seeds; "
        "not a multiple-comparison-adjusted confirmatory result",
    }


def topk_numpy(documents, queries, k):
    # Float64 exhaustive scores independently check the native float32 FAISS path.
    scores = queries.astype(np.float64) @ documents.astype(np.float64).T
    rows = np.empty((len(queries), k), dtype=int)
    ids = np.arange(len(documents))
    for i, row in enumerate(scores):
        rows[i] = np.lexsort((ids, -row))[:k]
    return rows, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrain", action="store_true", help="Refit PCA/RP/AE128 seed42")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = read_json(args.study / "study.json")
    if study["config"]["experiment"]["split"] != "dev":
        raise ValueError("This exploratory audit only reads dev; keep final test untouched")
    cache = Path(study["embedding_cache"])
    validate_cache(cache)
    settings = study["config"]["data"]
    dataset = load_dataset(Path(settings["directory"]) / settings["name"], settings)
    doc_ids = read_json(cache / "doc_ids.json")
    query_ids = read_json(cache / "query_ids.json")
    qids = sorted(dataset.qrels["dev"])
    qrels = dataset.qrels["dev"]
    positions = {qid: i for i, qid in enumerate(query_ids)}
    x = unit_vectors(np.load(cache / "documents.npy", allow_pickle=False))
    q = unit_vectors(
        np.load(cache / "queries.npy", allow_pickle=False)[[positions[qid] for qid in qids]]
    )
    results, individual, replay = {}, {}, []
    models = {}
    threads = study["config"]["experiment"]["threads"]
    faiss.omp_set_num_threads(threads)
    ks = study["config"]["experiment"]["ks"]
    with threadpool_limits(limits=threads, user_api="blas"):
        for path in sorted(args.study.glob("*/result.json")):
            folder, name = path.parent, path.parent.name
            result = read_json(path)
            for file, expected in result["files"].items():
                assert sha256(folder / file) == expected, (name, file)
            model = Compressor.load(folder)
            models[name] = model
            xz = unit_vectors(model.transform(x))
            qz = unit_vectors(model.transform(q))
            index = faiss.read_index(str(folder / "index.faiss"))
            stored = index.reconstruct_n(0, index.ntotal)
            np.testing.assert_allclose(stored, unit_vectors(xz), atol=2e-7)
            saved_rows, saved_scores = [], []
            ranking = {}
            for line in (folder / "ranking.jsonl").read_text().splitlines():
                row = json.loads(line)
                ranking[row["query_id"]] = row["document_ids"]
                saved_rows.append([doc_ids.index(did) for did in row["document_ids"]])
                saved_scores.append(row["scores"])
            saved_rows, saved_scores = np.array(saved_rows), np.array(saved_scores)
            assert list(ranking) == qids
            scalar = independent_metrics(qrels, ranking, ks)
            existing = read_json(folder / "per_query.json")
            for qid in qids:
                for metric, value in scalar[qid].items():
                    np.testing.assert_allclose(existing[qid][metric], value, atol=1e-12)
            for metric in result["metrics"]:
                np.testing.assert_allclose(
                    result["metrics"][metric],
                    np.mean([v[metric] for v in scalar.values()]),
                    atol=1e-12,
                )
            rows, exhaustive = topk_numpy(stored, unit_vectors(qz), saved_rows.shape[1])
            rescored = np.take_along_axis(exhaustive, saved_rows, axis=1)
            np.testing.assert_allclose(saved_scores, rescored, atol=2e-6, rtol=1e-6)
            reference_scores = np.take_along_axis(exhaustive, rows, axis=1)
            max_rank_score_gap = float(np.max(np.abs(reference_scores - rescored)))
            assert max_rank_score_gap < 2e-6, (name, max_rank_score_gap)
            float64_rankings = {
                qid: [doc_ids[row] for row in rows[i]] for i, qid in enumerate(qids)
            }
            alternative, _ = evaluate(qrels, float64_rankings, ks)
            replay.append(
                {
                    "run": name,
                    "artifact_checksums": "pass",
                    "saved_index_matches_transform": True,
                    "scalar_metrics": "all per-query and aggregate values match",
                    "float64_top10_identical_queries": int(
                        np.all(rows[:, :10] == saved_rows[:, :10], axis=1).sum()
                    ),
                    "queries": len(qids),
                    "max_rank_score_gap": max_rank_score_gap,
                    "float64_ndcg@10_delta": alternative["ndcg@10"] - result["metrics"]["ndcg@10"],
                }
            )
            results[name], individual[name] = result, scalar
            print("Verified", name, flush=True)

        # PCA reconstruction defines the floor for centered linear rank-d approximation.
        centered = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
        diagnostics = []
        for name, model in models.items():
            if model.method != "ae":
                continue
            with np.load(args.study / name / "training_decoder.npz", allow_pickle=False) as f:
                decoder = f["decoder"].astype(np.float64)
            w = model.arrays["encoder"].astype(np.float64)
            reconstruction = centered @ w.T @ decoder.T
            ae_error = float(np.mean(np.sum((reconstruction - centered) ** 2, axis=1)))
            pca = models[f"pca-{model.dimension}d-seed0"]
            wp = pca.arrays["encoder"].astype(np.float64)
            pc_reconstruction = centered @ wp.T @ wp
            floor = float(np.mean(np.sum((pc_reconstruction - centered) ** 2, axis=1)))
            _, singular, orthobasis = np.linalg.svd(w, full_matrices=False)
            span_cosines = np.linalg.svd(wp @ orthobasis.T, compute_uv=False)
            zdocs = unit_vectors(centered @ orthobasis.T)
            zquery = unit_vectors((q - model.arrays["mean"]) @ orthobasis.T)
            rows, _ = topk_numpy(zdocs, zquery, max(ks))
            scores, _ = evaluate(
                qrels, {qid: [doc_ids[row] for row in rows[i]] for i, qid in enumerate(qids)}, ks
            )
            history = model.info["history"]
            diagnostics.append(
                {
                    "run": name,
                    "dimension": model.dimension,
                    "seed": model.seed,
                    "full_corpus_reconstruction": ae_error,
                    "pca_reconstruction_floor": floor,
                    "reconstruction_ratio_to_pca": ae_error / floor,
                    "encoder_condition_number": float(singular[0] / singular[-1]),
                    "mean_squared_subspace_cosine": float(np.mean(span_cosines**2)),
                    "raw_ae_ndcg@10": results[name]["metrics"]["ndcg@10"],
                    "orthonormalized_rowspace_ndcg@10": scores["ndcg@10"],
                    "pca_ndcg@10": results[f"pca-{model.dimension}d-seed0"]["metrics"]["ndcg@10"],
                    "last_probe_reduction_fraction": (
                        history[-2]["corpus_probe_reconstruction"]
                        - history[-1]["corpus_probe_reconstruction"]
                    )
                    / history[-2]["corpus_probe_reconstruction"],
                    "warning": "Orthonormalized rowspace is a diagnostic transformation, "
                    "not a replacement of the measured vanilla AE baseline",
                }
            )
        comparisons = []
        groups = defaultdict(list)
        for name, result in results.items():
            if result["method"] == "ae":
                groups[result["dimension"]].append(name)
        for dimension, names in sorted(groups.items()):
            pca = individual[f"pca-{dimension}d-seed0"]
            differences = [
                [individual[name][qid]["ndcg@10"] - pca[qid]["ndcg@10"] for qid in qids]
                for name in names
            ]
            comparisons.append(
                {
                    "dimension": dimension,
                    "candidate_runs": names,
                    **seed_query_bootstrap(differences),
                }
            )

        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered_docs, centered_queries = unit_vectors(x - mean), unit_vectors(q - mean)
        centered_rows, _ = topk_numpy(centered_docs, centered_queries, max(ks))
        centered_quality, _ = evaluate(
            qrels,
            {qid: [doc_ids[row] for row in centered_rows[i]] for i, qid in enumerate(qids)},
            ks,
        )
        source_original = args.study / f"original-{x.shape[1]}d-seed0" / "ranking.jsonl"
        original_rows = np.array(
            [
                [doc_ids.index(did) for did in json.loads(line)["document_ids"]]
                for line in source_original.read_text().splitlines()
            ]
        )
        centered_quality["query_doc_overlap@10"] = float(
            overlap(original_rows, centered_rows, 10).mean()
        )
        diagnostic_output = {
            "centering_without_dimension_reduction": centered_quality,
            "ae": diagnostics,
            "ae_vs_pca_seed_query_bootstrap": comparisons,
        }
        write_json(args.output / "numerical-replay.json", replay)
        write_json(args.output / "diagnostics.json", diagnostic_output)
        write_json(
            args.output / "provenance.json",
            {
                "source_study": str(args.study),
                "source_study_manifest_sha256": sha256(args.study / "study.json"),
                "source_code_hash_at_original_run": study["source_sha256"],
                "query_split": "dev only",
                "test_used": False,
                "ranking_tolerance": "Float64 independent exhaustive score gap <2e-6",
            },
        )

        if args.retrain:
            reproductions = []
            for method in ["pca", "rp", "ae"]:
                seed = 0 if method == "pca" else 42
                name = f"{method}-128d-seed{seed}"
                fresh = Compressor(method, 128, seed).fit(x, study["config"]["ae"])
                original = models[name]
                maximum = max(
                    float(np.max(np.abs(fresh.arrays[key] - original.arrays[key])))
                    for key in original.arrays
                )
                np.testing.assert_allclose(fresh.transform(q), original.transform(q), atol=1e-7)
                reproductions.append(
                    {
                        "run": name,
                        "maximum_saved_parameter_difference": maximum,
                        "fit_config_fingerprint": fingerprint(study["config"]["ae"]),
                        "query_transform_reproduces": True,
                    }
                )
            write_json(args.output / "retraining.json", reproductions)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

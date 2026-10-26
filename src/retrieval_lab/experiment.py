"""Shared corpus-only fitting, exact retrieval, metrics and run provenance."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .common import (
    environment,
    fingerprint,
    read_json,
    sha256,
    source_digest,
    unit_vectors,
    write_json,
)
from .compressors import Compressor
from .data import Dataset
from .embedding import validate_cache
from .metrics import evaluate, overlap, paired_bootstrap
from .search import ExactIndex


def run(config: dict, dataset: Dataset, cache: Path) -> Path:
    embedding_manifest = validate_cache(cache)
    x = unit_vectors(np.load(cache / "documents.npy", allow_pickle=False))
    all_queries = np.load(cache / "queries.npy", allow_pickle=False)
    doc_ids = read_json(cache / "doc_ids.json")
    cached_qids = read_json(cache / "query_ids.json")
    if doc_ids != sorted(dataset.corpus) or len(x) != len(doc_ids):
        raise ValueError("Cached document order/count differs from corpus")
    exp = config["experiment"]
    qrels = dataset.qrels[exp["split"]]
    qids = sorted(qrels)
    positions = {qid: i for i, qid in enumerate(cached_qids)}
    q = unit_vectors(all_queries[[positions[qid] for qid in qids]])
    if max(exp["dimensions"]) > x.shape[1]:
        raise ValueError("Requested compression dimension exceeds model output dimension")
    if max(exp["ks"]) > len(x):
        raise ValueError("Metric cutoff exceeds corpus size; use smaller ks")
    if len(x) < 11:
        raise ValueError("The overlap@10 protocol requires at least 11 corpus documents")
    observed_environment = environment()
    execution = {
        key: observed_environment[key]
        for key in ["python", "platform", "machine", "hardware", "packages"]
    }
    spec = {
        "config": config,
        "embedding_files": embedding_manifest["files"],
        "dataset": dataset.manifest,
        "source_sha256": source_digest(),
        "execution_environment": execution,
    }
    study_id = fingerprint(spec)[:12]
    root = Path(exp["output"]) / exp["split"] / study_id
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "study.json").exists():
        write_json(
            root / "study.json",
            {
                **spec,
                "environment": observed_environment,
                "study_id": study_id,
                "embedding_cache": str(cache.resolve()),
                "query_ids": qids,
                "document_ids_sha256": fingerprint(doc_ids),
                "protocol": "corpus-adaptive; no relevance labels used for compressor fitting",
                "query_document_same_string_id": "retained; independent ID namespaces",
            },
        )
    threads = exp["threads"]
    search_depth = max(10, max(exp["ks"]))
    with threadpool_limits(limits=threads, user_api="blas"):
        original_index = ExactIndex(x, threads)
        original_scores, original_rows = original_index.search(q, search_depth)
        rankings = {qid: [doc_ids[row] for row in original_rows[i]] for i, qid in enumerate(qids)}
        _, reference = evaluate(qrels, rankings, exp["ks"])
        anchors = np.sort(
            np.random.default_rng(2026).choice(
                len(x), min(exp["geometry_anchors"], len(x)), replace=False
            )
        )
        geometry_k = min(10, len(x) - 1)
        _, original_doc_rows = original_index.search(x[anchors], geometry_k, anchors)
        jobs = []
        for method in exp["methods"]:
            dims = [x.shape[1]] if method == "original" else exp["dimensions"]
            seeds = [0] if method in {"original", "pca"} else exp["seeds"]
            jobs.extend((method, dim, seed) for dim in dims for seed in seeds)
        for method, dim, seed in jobs:
            folder = root / f"{method}-{dim}d-seed{seed}"
            if (folder / "result.json").exists():
                saved = read_json(folder / "result.json")
                if all(sha256(folder / name) == value for name, value in saved["files"].items()):
                    print(f"Reusing complete run {folder.name}", flush=True)
                    continue
                raise ValueError(f"Existing run failed checksum validation: {folder}")
            print(f"Running {method} {x.shape[1]}->{dim}, seed={seed}", flush=True)
            compressor = Compressor(method, dim, seed).fit(x, config["ae"])
            compressor.save(folder)
            start = time.perf_counter()
            xz = unit_vectors(compressor.transform(x))
            doc_transform = time.perf_counter() - start
            start = time.perf_counter()
            qz = unit_vectors(compressor.transform(q))
            query_transform = time.perf_counter() - start
            start = time.perf_counter()
            index = ExactIndex(xz, threads)
            index_seconds = time.perf_counter() - start
            scores, rows = index.search(qz, search_depth)
            rankings = {qid: [doc_ids[row] for row in rows[i]] for i, qid in enumerate(qids)}
            metrics, per_query = evaluate(qrels, rankings, exp["ks"])
            query_overlap = overlap(original_rows, rows, 10)
            _, doc_rows = index.search(xz[anchors], geometry_k, anchors)
            doc_overlap = overlap(original_doc_rows, doc_rows, geometry_k)
            for i, qid in enumerate(qids):
                per_query[qid]["query_doc_overlap@10"] = float(query_overlap[i])
            write_json(folder / "per_query.json", per_query)
            with (folder / "ranking.jsonl").open("w") as file:
                for i, qid in enumerate(qids):
                    file.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "document_ids": rankings[qid],
                                "scores": scores[i].tolist(),
                            }
                        )
                        + "\n"
                    )
            write_json(
                folder / "geometry.json",
                {
                    "anchor_document_ids": [doc_ids[row] for row in anchors],
                    "doc_doc_overlap": doc_overlap.tolist(),
                    "k": geometry_k,
                    "self_match": "excluded by row ID, not assumed to be first neighbor",
                },
            )
            faiss.write_index(index.index, str(folder / "index.faiss"))
            timing = index.benchmark(
                qz, max(exp["ks"]), exp["latency_queries"], exp["latency_repeats"]
            )
            timing.update(
                {
                    "fit_seconds": compressor.info["fit_seconds"],
                    "document_transform_seconds": doc_transform,
                    "query_batch_transform_seconds": query_transform,
                    "query_batch_transform_ms_per_query": query_transform / len(q) * 1000,
                    "index_build_seconds": index_seconds,
                }
            )
            result = {
                "dataset": dataset.name,
                "split": exp["split"],
                "method": method,
                "dimension": dim,
                "seed": seed,
                "documents": len(x),
                "queries": len(q),
                "metrics": metrics,
                "timing": timing,
                "query_doc_overlap@10": float(query_overlap.mean()),
                "doc_doc_overlap@10": float(doc_overlap.mean()),
                "storage": {
                    "vector_payload_bytes": int(xz.nbytes),
                    "index_file_bytes": (folder / "index.faiss").stat().st_size,
                    "compressor_file_bytes": (folder / "compressor.npz").stat().st_size,
                    "training_decoder_file_bytes": (
                        (folder / "training_decoder.npz").stat().st_size if method == "ae" else 0
                    ),
                },
                "paired_vs_original": paired_bootstrap(
                    reference,
                    per_query,
                    "ndcg@10" if 10 in exp["ks"] else f"ndcg@{exp['ks'][0]}",
                    exp["bootstrap_samples"],
                ),
                "study_id": study_id,
                "files": {
                    name: sha256(folder / name)
                    for name in [
                        "compressor.npz",
                        "compressor.json",
                        "per_query.json",
                        "ranking.jsonl",
                        "geometry.json",
                        "index.faiss",
                    ]
                },
            }
            write_json(folder / "result.json", result)
            print(f"  {metrics}", flush=True)
        # Baseline reference is used even when not requested as a saved run.
        del original_index, original_scores
    return root


def demo(output: str) -> Path:
    """Synthetic executable integration check; never evidence of BEIR performance."""
    from .config import DEFAULTS

    config = copy.deepcopy(DEFAULTS)
    config["experiment"].update(
        {
            "output": output,
            "dimensions": [16, 8],
            "seeds": [17, 42],
            "ks": [10, 20],
            "geometry_anchors": 32,
            "latency_queries": 10,
            "latency_repeats": 1,
            "bootstrap_samples": 200,
        }
    )
    config["ae"].update({"steps": 200, "batch_size": 64, "log_every": 100})
    rng = np.random.default_rng(2026)
    basis = rng.normal(size=(8, 32))
    latent = rng.normal(size=(160, 8))
    x = unit_vectors(latent @ basis + 0.02 * rng.normal(size=(160, 32)))
    q = unit_vectors(x[:24] + 0.01 * rng.normal(size=(24, 32)))
    doc_ids = [f"doc-{i:04}" for i in range(len(x))]
    qids = [f"query-{i:04}" for i in range(len(q))]
    dataset = Dataset(
        "SYNTHETIC",
        {did: {} for did in doc_ids},
        dict.fromkeys(qids, "synthetic"),
        {"dev": {qid: {doc_ids[i]: 1} for i, qid in enumerate(qids)}},
        {"name": "SYNTHETIC", "seed": 2026, "not_a_real_benchmark": True},
    )
    cache = Path(output) / "synthetic-cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "documents.npy", x)
    np.save(cache / "queries.npy", q)
    write_json(cache / "doc_ids.json", doc_ids)
    write_json(cache / "query_ids.json", qids)
    write_json(
        cache / "manifest.json",
        {
            "files": {
                name: sha256(cache / name)
                for name in ["documents.npy", "queries.npy", "doc_ids.json", "query_ids.json"]
            }
        },
    )
    return run(config, dataset, cache)

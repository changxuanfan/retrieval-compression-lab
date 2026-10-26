"""Query-macro IR metrics aligned with trec_eval's linear relevance gain."""

from __future__ import annotations

import numpy as np


def evaluate(qrels: dict, rankings: dict, ks: list[int]) -> tuple[dict, dict]:
    if not qrels:
        raise ValueError("Cannot evaluate an empty query set")
    per_query = {}
    for qid, rels in qrels.items():
        documents = rankings.get(qid, [])
        if len(set(documents)) != len(documents):
            raise ValueError(f"Duplicate retrieved document for query {qid}")
        positives = {did for did, grade in rels.items() if grade > 0}
        ideal = sorted((grade for grade in rels.values() if grade > 0), reverse=True)
        scores = {}
        for k in ks:
            top = documents[:k]
            discount = 1 / np.log2(np.arange(2, k + 2))
            gains = np.array([max(0, rels.get(did, 0)) for did in top], dtype=float)
            dcg = float(gains @ discount[: len(gains)])
            best = np.array(ideal[:k], dtype=float)
            idcg = float(best @ discount[: len(best)])
            scores[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
            scores[f"recall@{k}"] = len(set(top) & positives) / len(positives) if positives else 0.0
            first = next((i + 1 for i, did in enumerate(top) if did in positives), None)
            scores[f"mrr@{k}"] = 1 / first if first is not None else 0.0
        per_query[qid] = scores
    aggregate = {
        key: float(np.mean([v[key] for v in per_query.values()]))
        for key in next(iter(per_query.values()))
    }
    return aggregate, per_query


def overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    k = min(k, reference.shape[1], candidate.shape[1])
    if k == 0:
        raise ValueError("No neighbors to compare")
    return np.array(
        [len(set(a[:k]) & set(b[:k])) / k for a, b in zip(reference, candidate, strict=True)],
        dtype=float,
    )


def paired_bootstrap(
    reference: dict, candidate: dict, metric: str, samples: int = 2000, seed: int = 2026
) -> dict:
    if set(reference) != set(candidate) or not reference:
        raise ValueError("Paired bootstrap requires identical nonempty query IDs")
    ids = sorted(reference)
    differences = np.array([candidate[qid][metric] - reference[qid][metric] for qid in ids])
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    # Small batches avoid allocating samples x many-query matrices.
    for start in range(0, samples, 128):
        end = min(start + 128, samples)
        rows = rng.integers(len(ids), size=(end - start, len(ids)))
        means[start:end] = differences[rows].mean(axis=1)
    return {
        "metric": metric,
        "delta": float(differences.mean()),
        "query_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "samples": samples,
        "seed": seed,
        "interpretation": "conditional on this fitted seed; not seed uncertainty",
    }

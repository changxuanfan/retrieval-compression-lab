"""Exact full-corpus cosine search and CPU-only search latency measurement."""

from __future__ import annotations

import time

import faiss
import numpy as np

from .common import unit_vectors


class ExactIndex:
    """Corpus rows must be sorted by document ID for canonical score ties."""

    def __init__(self, vectors: np.ndarray, threads: int = 4):
        faiss.omp_set_num_threads(threads)
        self.vectors = unit_vectors(vectors)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, queries: np.ndarray, k: int, exclude_rows=None):
        queries = unit_vectors(queries)
        n = len(self.vectors)
        k = min(k, n - (1 if exclude_rows is not None else 0))
        if k < 1:
            raise ValueError("No searchable documents after exclusions")
        fetch = min(n, k + 2)
        scores, rows = self.index.search(queries, fetch)
        result_rows, result_scores = [], []
        for i in range(len(queries)):
            current_scores, current_rows = scores[i], rows[i]
            current_fetch = fetch
            while True:
                keep = current_rows >= 0
                if exclude_rows is not None:
                    keep &= current_rows != exclude_rows[i]
                valid_scores, valid_rows = current_scores[keep], current_rows[keep]
                order = np.lexsort((valid_rows, -valid_scores))
                valid_scores, valid_rows = valid_scores[order], valid_rows[order]
                # Expand cutoff ties, including any tied candidates outside the initial top-k.
                if current_fetch == n or (
                    len(valid_scores) > k and valid_scores[k - 1] > valid_scores[-1]
                ):
                    result_rows.append(valid_rows[:k])
                    result_scores.append(valid_scores[:k])
                    break
                current_fetch = min(n, current_fetch * 2)
                expanded_scores, expanded_rows = self.index.search(
                    queries[i : i + 1], current_fetch
                )
                current_scores, current_rows = expanded_scores[0], expanded_rows[0]
        return np.array(result_scores), np.array(result_rows)

    def benchmark(self, queries: np.ndarray, k: int, count: int, repeats: int) -> dict:
        queries = unit_vectors(queries[:count])
        self.search(queries[: min(10, len(queries))], k)
        latencies = []
        for _ in range(repeats):
            for query in queries:
                start = time.perf_counter()
                self.search(query[None, :], k)
                latencies.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        self.search(queries, k)
        seconds = time.perf_counter() - start
        return {
            "search_p50_ms": float(np.median(latencies)),
            "search_p95_ms": float(np.quantile(latencies, 0.95)),
            "batch_search_queries_per_second": len(queries) / seconds,
            "timed_queries": len(queries),
            "repeats": repeats,
            "scope": "CPU exact search wrapper; excludes model encoding and projection; "
            "includes normalization, result ordering and tie expansion",
        }

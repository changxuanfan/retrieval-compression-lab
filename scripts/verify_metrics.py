"""Cross-check a real study's graded nDCG against independent sklearn code.

Uses full-corpus qrels for IDCG and encodes saved order as strictly decreasing
scores so that the check follows the repository's documented tie convention.
Requires the development extra; does not download data or models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

from retrieval_lab.common import read_json
from retrieval_lab.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study = read_json(args.study / "study.json")
    settings = study["config"]["data"]
    dataset = load_dataset(Path(settings["directory"]) / settings["name"], settings)
    ids = sorted(dataset.corpus)
    columns = {did: i for i, did in enumerate(ids)}
    qrels = dataset.qrels[study["config"]["experiment"]["split"]]
    qids = sorted(qrels)
    labels = np.zeros((len(qids), len(ids)))
    for i, qid in enumerate(qids):
        for did, grade in qrels[qid].items():
            labels[i, columns[did]] = grade
    for path in sorted(args.study.glob("*/result.json")):
        scores = np.zeros_like(labels)
        ranked = {}
        with (path.parent / "ranking.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                ranked[row["query_id"]] = row["document_ids"]
        for i, qid in enumerate(qids):
            docs = ranked[qid]
            for rank, did in enumerate(docs):
                scores[i, columns[did]] = len(docs) - rank
        result = read_json(path)
        for k in study["config"]["experiment"]["ks"]:
            expected = ndcg_score(labels, scores, k=k, ignore_ties=True)
            np.testing.assert_allclose(result["metrics"][f"ndcg@{k}"], expected, atol=1e-12)
        print(f"PASS: {path.parent.name}; full-qrels sklearn nDCG cross-check")


if __name__ == "__main__":
    main()

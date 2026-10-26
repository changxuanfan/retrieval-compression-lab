"""Measure preprocessing truncation using the exact cached tokenizer revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from retrieval_lab.common import read_json, write_json
from retrieval_lab.data import load_dataset


def describe(lengths, limit):
    array = np.array(lengths)
    return {
        "count": len(array),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "over_limit_count": int(np.sum(array > limit)),
        "over_limit_fraction": float(np.mean(array > limit)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = read_json(args.study / "study.json")
    if study["config"]["experiment"]["split"] != "dev":
        raise ValueError("This exploratory preprocessing audit only inspects dev queries")
    metadata = read_json(Path(study["embedding_cache"]) / "manifest.json")
    settings = metadata["spec"]["embedding"]
    data_settings = study["config"]["data"]
    dataset = load_dataset(Path(data_settings["directory"]) / data_settings["name"], data_settings)
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=metadata["resolved_model_revision"], local_files_only=True
    )
    doc_ids = sorted(dataset.corpus)
    texts = [
        settings["document_prefix"]
        + (dataset.corpus[did].get("title", "") + " " + dataset.corpus[did]["text"]).strip()
        for did in doc_ids
    ]
    lengths = tokenizer(texts, truncation=False, padding=False, return_length=True, verbose=False)[
        "length"
    ]
    qids = sorted(dataset.qrels["dev"])
    query_texts = [settings["query_prefix"] + dataset.queries[qid] for qid in qids]
    query_lengths = tokenizer(
        query_texts, truncation=False, padding=False, return_length=True, verbose=False
    )["length"]
    limit = settings["max_length"]
    by_doc = dict(zip(doc_ids, lengths, strict=True))
    relevant = {
        did
        for judgments in dataset.qrels["dev"].values()
        for did, grade in judgments.items()
        if grade > 0
    }
    write_json(
        args.output,
        {
            "model": settings["model"],
            "revision": metadata["resolved_model_revision"],
            "limit": limit,
            "special_tokens_and_prefixes_included": True,
            "corpus": describe(lengths, limit),
            "dev_queries": describe(query_lengths, limit),
            "unique_dev_relevant_documents": describe([by_doc[did] for did in relevant], limit),
            "doc_token_lengths": by_doc,
            "interpretation": "Preprocessing limitation shared by all four methods; "
            "not causal proof of retrieval failure; test queries untouched",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

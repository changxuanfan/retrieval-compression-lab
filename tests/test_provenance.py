import copy
import json

import numpy as np
import pytest

from retrieval_lab.common import read_json, sha256, unit_vectors, write_json
from retrieval_lab.config import DEFAULTS
from retrieval_lab.data import Dataset
from retrieval_lab.experiment import run


@pytest.fixture
def small_study(tmp_path):
    config = copy.deepcopy(DEFAULTS)
    config["experiment"].update(
        {
            "output": str(tmp_path / "runs"),
            "methods": ["original"],
            "ks": [1, 3],
            "dimensions": [4],
            "geometry_anchors": 5,
            "latency_queries": 2,
            "latency_repeats": 1,
            "bootstrap_samples": 20,
        }
    )
    x = unit_vectors(np.random.default_rng(42).normal(size=(30, 8)))
    doc_ids = [f"d{i:03}" for i in range(30)]
    qids = ["q0", "q1"]
    dataset = Dataset(
        "fixture",
        dict.fromkeys(doc_ids, {}),
        dict.fromkeys(qids, "query"),
        {"dev": {"q0": {doc_ids[0]: 1}, "q1": {doc_ids[1]: 1}}},
        {},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "documents.npy", x)
    np.save(cache / "queries.npy", x[:2])
    write_json(cache / "doc_ids.json", doc_ids)
    write_json(cache / "query_ids.json", qids)
    write_json(
        cache / "manifest.json",
        {
            "files": {
                name: sha256(cache / name)
                for name in [
                    "documents.npy",
                    "queries.npy",
                    "doc_ids.json",
                    "query_ids.json",
                ]
            }
        },
    )
    return config, dataset, cache


def test_short_metric_cutoffs_still_retrieve_ten_neighbors(small_study):
    root = run(*small_study)
    folder = root / "original-8d-seed0"
    ranking = json.loads((folder / "ranking.jsonl").read_text().splitlines()[0])
    assert len(ranking["document_ids"]) == 10
    assert read_json(folder / "result.json")["query_doc_overlap@10"] == 1


def test_environment_changes_create_new_studies_and_reuse_preserves_metadata(
    small_study, monkeypatch
):
    import retrieval_lab.experiment as experiment

    original_environment = experiment.environment
    first = run(*small_study)
    original_manifest = (first / "study.json").read_bytes()
    assert run(*small_study) == first
    assert (first / "study.json").read_bytes() == original_manifest

    def other_machine():
        value = original_environment()
        value["hardware"]["cpu_model"] = "a different machine"
        return value

    monkeypatch.setattr(experiment, "environment", other_machine)
    second = run(*small_study)
    assert first != second
    assert (first / "study.json").read_bytes() == original_manifest

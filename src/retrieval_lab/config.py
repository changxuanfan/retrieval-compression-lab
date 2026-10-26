"""Explicit TOML experiment configuration; paths resolve relative to repository root."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULTS = {
    "data": {"name": "scifact", "directory": "data", "split_seed": 2026, "dev_fraction": 0.2},
    "embedding": {
        "model": "intfloat/e5-base-v2",
        "revision": "main",
        "directory": "cache/embeddings",
        "device": "auto",
        "batch_size": 16,
        "max_length": 512,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "experiment": {
        "output": "runs/scifact",
        "split": "dev",
        "methods": ["original", "pca", "rp", "ae"],
        "dimensions": [384, 256, 128, 64],
        "seeds": [17, 42, 2026],
        "ks": [10, 100],
        "threads": 4,
        "geometry_anchors": 256,
        "latency_queries": 100,
        "latency_repeats": 3,
        "bootstrap_samples": 2000,
    },
    "ae": {
        "threads": 4,
        "steps": 3000,
        "batch_size": 128,
        "lr": 0.001,
        "min_lr": 0.00001,
        "device": "cpu",
        "log_every": 100,
    },
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    supplied = tomllib.loads(path.read_text())
    config = copy.deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if section not in config or set(values) - set(config[section]):
            raise ValueError(f"Unknown configuration section/keys: {section}")
        config[section].update(values)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config["data"]["name"] not in {"scifact", "nfcorpus", "fiqa"}:
        raise ValueError("Supported datasets: scifact, nfcorpus, fiqa")
    if not 0 < config["data"]["dev_fraction"] < 1:
        raise ValueError("dev_fraction must be between zero and one")
    exp, ae = config["experiment"], config["ae"]
    if exp["split"] not in {"train", "dev", "test"}:
        raise ValueError("split must be train/dev/test")
    methods = exp["methods"]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be a nonempty list without duplicates")
    if set(methods) - {"original", "pca", "rp", "ae"}:
        raise ValueError("Only original, pca, rp, ae are implemented; RAE/QRAE are planning only")
    for field in ["dimensions", "ks"]:
        values = exp[field]
        if not values or len(set(values)) != len(values) or any(v < 1 for v in values):
            raise ValueError(f"{field} must contain distinct positive integers")
    if not exp["seeds"] or len(set(exp["seeds"])) != len(exp["seeds"]):
        raise ValueError("seeds must contain distinct integers")
    if any(v < 0 for v in exp["seeds"]):
        raise ValueError("seeds must be nonnegative")
    for key in [
        "threads",
        "geometry_anchors",
        "latency_queries",
        "latency_repeats",
        "bootstrap_samples",
    ]:
        if exp[key] < 1:
            raise ValueError(f"{key} must be positive")
    if ae["steps"] < 1 or ae["batch_size"] < 1 or ae["log_every"] < 1:
        raise ValueError("AE steps, batch_size, log_every must be positive")
    if not 0 < ae["min_lr"] <= ae["lr"]:
        raise ValueError("Require 0 < min_lr <= lr")

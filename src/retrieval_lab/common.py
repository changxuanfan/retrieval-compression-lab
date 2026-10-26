"""Small shared serialization, provenance and numeric utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_vectors(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Zero or collapsed embedding: cannot compute cosine similarity")
    return np.ascontiguousarray(x / norm, dtype=np.float32)


def source_digest() -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def environment() -> dict:
    versions = {}
    for package in [
        "numpy",
        "torch",
        "sentence-transformers",
        "huggingface-hub",
        "faiss-cpu",
        "matplotlib",
        "threadpoolctl",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    hardware = {"processor": platform.processor()}
    if platform.system() == "Darwin":
        for field, key in [
            ("machdep.cpu.brand_string", "cpu_model"),
            ("hw.memsize", "memory_bytes"),
        ]:
            try:
                hardware[key] = subprocess.check_output(
                    ["sysctl", "-n", field], text=True, stderr=subprocess.DEVNULL
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                hardware[key] = None
    elif hasattr(os, "sysconf"):
        try:
            hardware["memory_bytes"] = str(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": hardware,
        "packages": versions,
        "git_commit": commit,
        "source_sha256": source_digest(),
    }

"""A small real query path using the exact same frozen model and saved transform."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np

from .common import read_json, unit_vectors
from .compressors import Compressor
from .data import load_dataset
from .embedding import validate_cache


def query(folder: Path, text: str, k: int) -> None:
    if k < 1:
        raise ValueError("top-k must be positive")
    study = read_json(folder.parent / "study.json")
    cache = Path(study["embedding_cache"])
    metadata = validate_cache(cache)
    if "spec" not in metadata:
        raise ValueError("New-text query requires a real-model run, not the synthetic demo")
    with tempfile.TemporaryDirectory(prefix="retrieval-query-") as directory:
        target = Path(directory)
        from .common import write_json

        write_json(target / "request.json", {"metadata": metadata, "text": text})
        subprocess.run(
            [sys.executable, "-m", "retrieval_lab.worker", "query", str(target)], check=True
        )
        q = np.load(target / "query.npy", allow_pickle=False)
    compressor = Compressor.load(folder)
    qz = unit_vectors(compressor.transform(q))
    index = faiss.read_index(str(folder / "index.faiss"))
    scores, rows = index.search(qz, min(k, index.ntotal))
    ids = read_json(cache / "doc_ids.json")
    data_config = study["config"]["data"]
    dataset = load_dataset(Path(data_config["directory"]) / data_config["name"], data_config)
    for rank, (score, row) in enumerate(zip(scores[0], rows[0], strict=True), 1):
        document = dataset.corpus[ids[row]]
        print(f"{rank}. {ids[row]} | cosine={score:.5f} | {document.get('title', '')}")
        print(document["text"][:500], "\n")

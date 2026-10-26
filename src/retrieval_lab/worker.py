"""Torch-only worker: isolate macOS wheel OpenMP runtimes from FAISS.

Called internally; communicates only through numeric arrays and JSON, never pickle.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .compressors import Compressor, select_device


def main() -> None:
    operation, directory = sys.argv[1:]
    folder = Path(directory)
    request = read_json(folder / "request.json")
    if operation == "ae":
        x = np.load(folder / "inputs.npy", allow_pickle=False)
        model = Compressor("ae", request["dimension"], request["seed"])
        start = time.perf_counter()
        model._fit_ae_in_process(x, request["training"])
        model.info["training_worker_seconds"] = time.perf_counter() - start
        model.info["runtime_isolation"] = "Torch training worker; parent evaluates with FAISS"
        np.savez(folder / "trained.npz", **model.arrays)
        write_json(folder / "trained.json", model.info)
    elif operation == "query":
        from sentence_transformers import SentenceTransformer

        metadata = request["metadata"]
        settings = metadata["spec"]["embedding"]
        model = SentenceTransformer(
            settings["model"],
            revision=metadata["resolved_model_revision"],
            device=select_device(settings["device"]),
            trust_remote_code=False,
        )
        model.max_seq_length = settings["max_length"]
        vector = model.encode(
            [settings["query_prefix"] + request["text"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.save(folder / "query.npy", vector)
    else:
        raise ValueError(f"Unknown worker operation: {operation}")


if __name__ == "__main__":
    main()

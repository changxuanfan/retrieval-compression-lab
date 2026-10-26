"""Generate one immutable frozen-model cache shared by all four methods."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import environment, fingerprint, read_json, sha256, unit_vectors, write_json
from .compressors import select_device
from .data import Dataset


def _spec(config: dict, dataset: Dataset) -> dict:
    return {
        "embedding": config["embedding"],
        "dataset_manifest": dataset.manifest,
        "preprocessing": "title-space-text-strip; explicit prefixes once; ST pooling; L2; v1",
    }


def cache_path(config: dict, dataset: Dataset) -> Path:
    root = Path(config["embedding"]["directory"]) / dataset.name
    key = fingerprint(_spec(config, dataset))[:16]
    pointer = root / f"pointer-{key}.json"
    if not pointer.exists():
        raise FileNotFoundError("Embedding cache not found; run embed or pipeline first")
    folder = root / read_json(pointer)["folder"]
    validate_cache(folder)
    return folder


def validate_cache(folder: Path) -> dict:
    metadata = read_json(folder / "manifest.json")
    for name, checksum in metadata["files"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Embedding cache integrity failure: {name}")
    return metadata


def embed(config: dict, dataset: Dataset) -> Path:
    spec = _spec(config, dataset)
    settings = config["embedding"]
    root = Path(settings["directory"]) / dataset.name
    key = fingerprint(spec)[:16]
    pointer = root / f"pointer-{key}.json"
    if pointer.exists():
        folder = cache_path(config, dataset)
        print(f"Reusing verified embedding cache {folder}", flush=True)
        return folder

    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    revision = HfApi().model_info(settings["model"], revision=settings["revision"]).sha
    if not revision:
        raise ValueError("Could not resolve model to an immutable Hugging Face revision")
    folder = root / fingerprint({"spec": spec, "resolved_revision": revision})[:16]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        validate_cache(folder)
        write_json(pointer, {"folder": folder.name})
        return folder
    device = select_device(settings["device"])
    print(f"Loading {settings['model']} at {revision} on {device}", flush=True)
    model = SentenceTransformer(
        settings["model"], revision=revision, device=device, trust_remote_code=False
    )
    model.max_seq_length = settings["max_length"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    width = model.get_sentence_embedding_dimension()
    if width is None:
        raise ValueError("Model does not expose sentence embedding dimension")
    doc_ids = sorted(dataset.corpus)
    qids = sorted({qid for split in dataset.qrels.values() for qid in split})
    docs = [
        settings["document_prefix"]
        + (dataset.corpus[did].get("title", "") + " " + dataset.corpus[did]["text"]).strip()
        for did in doc_ids
    ]
    queries = [settings["query_prefix"] + dataset.queries[qid] for qid in qids]
    timings = {}
    for name, texts in [("documents", docs), ("queries", queries)]:
        start = time.perf_counter()
        partial = folder / f"{name}.part.npy"
        array = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(len(texts), width)
        )
        for offset in range(0, len(texts), 512):
            part = model.encode(
                texts[offset : offset + 512],
                batch_size=settings["batch_size"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            array[offset : offset + len(part)] = unit_vectors(part)
            array.flush()
            print(f"Encoded {name} {min(offset + 512, len(texts))}/{len(texts)}", flush=True)
        del array
        partial.replace(folder / f"{name}.npy")
        timings[f"{name}_encoding_seconds"] = time.perf_counter() - start
    write_json(folder / "doc_ids.json", doc_ids)
    write_json(folder / "query_ids.json", qids)
    metadata = {
        "spec": spec,
        "resolved_model_revision": revision,
        "actual_device": device,
        "dimension": width,
        "documents": len(docs),
        "queries": len(queries),
        "timings": timings,
        "environment": environment(),
        "encoding_timing_scope": "batched encoding incl tokenization, normalization, "
        "array writes; excludes model download/loading",
        "files": {
            name: sha256(folder / name)
            for name in ["documents.npy", "queries.npy", "doc_ids.json", "query_ids.json"]
        },
    }
    write_json(folder / "manifest.json", metadata)
    write_json(pointer, {"folder": folder.name})
    return folder

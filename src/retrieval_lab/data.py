"""Read official BEIR JSONL/TSV; no retrieval framework is copied or imported."""

from __future__ import annotations

import csv
import json
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import read_json, sha256, write_json

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass
class Dataset:
    name: str
    corpus: dict[str, dict]
    queries: dict[str, str]
    qrels: dict[str, dict[str, dict[str, int]]]
    manifest: dict


def extract_dataset(archive: Path, destination: Path, name: str) -> None:
    """Extract only this dataset's JSONL/TSV, rejecting traversal and symlinks."""
    with zipfile.ZipFile(archive) as file:
        for member in file.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe zip path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Zip symlinks are not accepted")
            if not path.parts or path.parts[0] != name:
                continue
            if member.is_dir() or path.suffix not in {".jsonl", ".tsv"}:
                continue
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with file.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def prepare(config: dict) -> Path:
    name = config["data"]["name"]
    root = Path(config["data"]["directory"])
    folder = root / name
    archive = root / "downloads" / f"{name}.zip"
    required = [
        folder / "corpus.jsonl",
        folder / "queries.jsonl",
        folder / "qrels/test.tsv",
        folder / "qrels/train.tsv",
    ]
    if not all(file.is_file() for file in required):
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            partial = archive.with_suffix(".zip.part")
            print(f"Downloading official BEIR {name} archive", flush=True)
            with urllib.request.urlopen(f"{BASE_URL}/{name}.zip", timeout=120) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
            partial.replace(archive)
        extract_dataset(archive, root, name)
    if not all(file.is_file() for file in required):
        raise ValueError(f"Incomplete BEIR dataset at {folder}")
    files = [folder / "corpus.jsonl", folder / "queries.jsonl"]
    files += sorted((folder / "qrels").glob("*.tsv"))
    manifest = {
        "name": name,
        "source": f"{BASE_URL}/{name}.zip",
        "archive_sha256": sha256(archive) if archive.exists() else None,
        "files": {str(file.relative_to(folder)): sha256(file) for file in files},
        "split_seed": config["data"]["split_seed"],
        "scifact_dev_fraction": config["data"]["dev_fraction"],
    }
    write_json(folder / "manifest.json", manifest)
    dataset = load_dataset(folder, config["data"])
    write_json(folder / "query_splits.json", {k: sorted(v) for k, v in dataset.qrels.items()})
    print(
        f"{name}: {len(dataset.corpus)} documents; query splits "
        f"{ {k: len(v) for k, v in dataset.qrels.items()} }",
        flush=True,
    )
    return folder


def _jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            key = str(row["_id"])
            if key in rows:
                raise ValueError(f"Duplicate ID {key!r} in {path}")
            rows[key] = row
    return rows


def load_dataset(folder: Path, data_config: dict) -> Dataset:
    corpus = _jsonl(folder / "corpus.jsonl")
    queries = {key: row["text"] for key, row in _jsonl(folder / "queries.jsonl").items()}
    splits = {}
    for path in sorted((folder / "qrels").glob("*.tsv")):
        qrels = {}
        with path.open() as file:
            for row in csv.DictReader(file, delimiter="\t"):
                qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
                if qid not in queries or did not in corpus:
                    raise ValueError(f"qrels refers to missing query/document: {qid}/{did}")
                if score < 0:
                    raise ValueError("Negative relevance grades are unsupported")
                rels = qrels.setdefault(qid, {})
                if did in rels and rels[did] != score:
                    raise ValueError("Conflicting duplicate relevance grade")
                rels[did] = score
        splits[path.stem] = qrels
    if "dev" not in splits:
        ids = sorted(splits["train"])
        np.random.default_rng(data_config["split_seed"]).shuffle(ids)
        count = max(1, round(len(ids) * data_config["dev_fraction"]))
        dev = set(ids[:count])
        splits["dev"] = {qid: splits["train"][qid] for qid in sorted(dev)}
        splits["train"] = {qid: rel for qid, rel in splits["train"].items() if qid not in dev}
    split_names = sorted(splits)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            if set(splits[left]) & set(splits[right]):
                raise ValueError(f"Query IDs overlap between {left} and {right}")
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if manifest:
        for file, expected in manifest["files"].items():
            if sha256(folder / file) != expected:
                raise ValueError(f"Dataset changed since preparation: {file}")
    return Dataset(data_config["name"], corpus, queries, splits, manifest)

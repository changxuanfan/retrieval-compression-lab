import io
import json
import zipfile

import pytest

from retrieval_lab.data import extract_dataset, load_dataset


def fixture_dataset(folder):
    folder.mkdir()
    (folder / "qrels").mkdir()
    for name, rows in [
        ("corpus", [{"_id": "d", "text": "document"}]),
        ("queries", [{"_id": str(i), "text": f"query {i}"} for i in range(11)]),
    ]:
        (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (folder / "qrels/train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"{i}\td\t1\n" for i in range(10))
    )
    (folder / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n10\td\t2\n")


def test_query_disjoint_split_is_reproducible_and_preserves_grades(tmp_path):
    folder = tmp_path / "scifact"
    fixture_dataset(folder)
    settings = {"name": "scifact", "split_seed": 42, "dev_fraction": 0.2}
    first = load_dataset(folder, settings)
    second = load_dataset(folder, settings)
    assert first.qrels == second.qrels
    assert len(first.qrels["train"]) == 8
    assert len(first.qrels["dev"]) == 2
    assert not set(first.qrels["train"]) & set(first.qrels["dev"])
    assert first.qrels["test"]["10"]["d"] == 2


def test_train_test_query_leakage_fails(tmp_path):
    folder = tmp_path / "scifact"
    fixture_dataset(folder)
    (folder / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n1\td\t1\n")
    with pytest.raises(ValueError, match="overlap"):
        load_dataset(folder, {"name": "scifact", "split_seed": 42, "dev_fraction": 0.2})


def test_zip_traversal_is_rejected(tmp_path):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("../outside.jsonl", "malicious")
    path = tmp_path / "archive.zip"
    path.write_bytes(memory.getvalue())
    with pytest.raises(ValueError, match="Unsafe"):
        extract_dataset(path, tmp_path, "scifact")

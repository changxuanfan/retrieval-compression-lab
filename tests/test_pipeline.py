import sys

from retrieval_lab.common import read_json
from retrieval_lab.experiment import demo
from retrieval_lab.reporting import write_report


def test_all_four_methods_share_one_offline_pipeline(tmp_path):
    torch_was_imported = "torch" in sys.modules
    root = demo(str(tmp_path))
    # Regresses the actual macOS FAISS/Torch runtime-conflict call pattern.
    assert ("torch" in sys.modules) == torch_was_imported
    results = [read_json(path) for path in root.glob("*/result.json")]
    assert len(results) == 11  # original + 2 PCA + 4 RP + 4 AE
    assert {row["method"] for row in results} == {"original", "pca", "rp", "ae"}
    assert all(row["dataset"] == "SYNTHETIC" for row in results)
    original = next(row for row in results if row["method"] == "original")
    assert original["metrics"]["ndcg@10"] == 1
    assert original["query_doc_overlap@10"] == 1
    assert original["doc_doc_overlap@10"] == 1
    assert write_report(root).exists()
    assert (root / "quality-memory.png").exists()

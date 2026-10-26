import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from retrieval_lab.metrics import evaluate, paired_bootstrap


def test_graded_ndcg_matches_independent_standard_without_ties():
    qrels = {"q": {"a": 2, "b": 1, "c": 1}}
    aggregate, _ = evaluate(qrels, {"q": ["b", "x", "a", "c"]}, [3])
    expected = ndcg_score(np.array([[2, 1, 1, 0]]), np.array([[2, 4, 1, 3]]), k=3)
    assert aggregate["ndcg@3"] == pytest.approx(expected)
    assert aggregate["recall@3"] == pytest.approx(2 / 3)
    assert aggregate["mrr@3"] == 1


def test_unreturned_queries_still_count_and_recall_is_not_hit_rate():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
    average, per_query = evaluate(qrels, {"q1": ["x", "a"]}, [2])
    assert average["recall@2"] == 0.25
    assert average["mrr@2"] == 0.25
    assert per_query["q2"]["ndcg@2"] == 0


def test_duplicate_retrieval_is_rejected_and_same_string_ids_retained():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate({"q": {"a": 1}}, {"q": ["a", "a"]}, [2])
    result, _ = evaluate({"same": {"same": 1}}, {"same": ["same"]}, [1])
    assert result["ndcg@1"] == 1


def test_bootstrap_paired_ids_and_degenerate_identity():
    scores = {str(i): {"ndcg@10": i / 10} for i in range(5)}
    result = paired_bootstrap(scores, scores, "ndcg@10", samples=100)
    assert result["query_bootstrap_95ci"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(scores, {"other": {}}, "ndcg@10")

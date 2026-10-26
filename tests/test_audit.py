import runpy
from pathlib import Path

import numpy as np
import pytest

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_study.py"))


def test_audit_scalar_metrics_include_missing_queries_and_full_qrels():
    scores = AUDIT["independent_metrics"](
        {"a": {"x": 2, "y": 1, "z": 1}, "b": {"w": 1}},
        {"a": ["unknown", "x"]},
        [2],
    )
    assert scores["a"]["recall@2"] == pytest.approx(1 / 3)
    assert scores["a"]["mrr@2"] == 0.5
    assert scores["b"]["ndcg@2"] == 0


def test_crossed_bootstrap_keeps_training_seed_uncertainty():
    # No query variability; a query-only bootstrap would wrongly collapse to 0.5.
    result = AUDIT["seed_query_bootstrap"]([[0, 0, 0], [1, 1, 1]], samples=2000)
    assert result["delta"] == 0.5
    assert result["seed_query_bootstrap_95ci"] == [0, 1]


def test_crossed_bootstrap_constant_difference_is_degenerate():
    result = AUDIT["seed_query_bootstrap"](np.full((3, 7), 0.02), samples=100)
    np.testing.assert_allclose(result["seed_query_bootstrap_95ci"], [0.02, 0.02])

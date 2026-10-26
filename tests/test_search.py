import numpy as np

from retrieval_lab.common import unit_vectors
from retrieval_lab.search import ExactIndex


def test_exact_search_matches_numpy_cosine_and_caps_small_corpus():
    rng = np.random.default_rng(42)
    x = unit_vectors(rng.normal(size=(60, 12)))
    q = unit_vectors(rng.normal(size=(5, 12)))
    index = ExactIndex(x, threads=2)
    scores, rows = index.search(q, 100)
    expected = np.argsort(-(q @ x.T), axis=1)
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(q @ x.T, rows, axis=1), atol=1e-6)


def test_cutoff_ties_expand_beyond_initial_topk_and_self_exclusion():
    x = np.ones((30, 4), dtype=np.float32)
    index = ExactIndex(x, threads=2)
    _, rows = index.search(x[:2], 3)
    np.testing.assert_array_equal(rows, [[0, 1, 2], [0, 1, 2]])
    _, rows = index.search(x[:2], 3, exclude_rows=np.array([0, 1]))
    np.testing.assert_array_equal(rows, [[1, 2, 3], [0, 2, 3]])

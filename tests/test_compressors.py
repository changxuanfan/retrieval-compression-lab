import numpy as np
import pytest
from sklearn.decomposition import PCA

from retrieval_lab.common import unit_vectors
from retrieval_lab.compressors import Compressor


def test_pca_matches_sklearn_geometry_on_out_of_sample_queries():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 12)) + 3
    queries = rng.normal(size=(7, 12)) - 1
    ours = Compressor("pca", 5).fit(x)
    standard = PCA(n_components=5, svd_solver="full").fit(x.astype(np.float32))
    z = ours.transform(queries)
    other = standard.transform(queries)
    np.testing.assert_allclose(z @ z.T, other @ other.T, rtol=1e-5, atol=1e-4)


def test_pca_shared_center_and_original_are_different():
    rng = np.random.default_rng(5)
    x = unit_vectors(rng.normal(size=(50, 6)) + 2)
    pca = Compressor("pca", 6).fit(x)
    z = unit_vectors(pca.transform(x))
    assert not np.allclose(z @ z.T, x @ x.T)
    np.testing.assert_array_equal(Compressor("original", 6).fit(x).transform(x), x)


def test_rp_seed_persistence_and_projection_distribution(tmp_path):
    x = np.ones((20, 512), dtype=np.float32)
    first = Compressor("rp", 256, 42).fit(x)
    same = Compressor("rp", 256, 42).fit(x)
    other = Compressor("rp", 256, 17).fit(x)
    np.testing.assert_array_equal(first.transform(x), same.transform(x))
    assert not np.array_equal(first.transform(x), other.transform(x))
    assert np.var(first.arrays["encoder"]) == pytest.approx(1 / 256, rel=0.03)
    first.save(tmp_path)
    np.testing.assert_array_equal(Compressor.load(tmp_path).transform(x), first.transform(x))


def test_autoencoder_learns_reconstruction_and_deployment_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    x = unit_vectors(rng.normal(size=(128, 3)) @ rng.normal(size=(3, 12)))
    config = {
        "steps": 350,
        "batch_size": 64,
        "lr": 0.01,
        "min_lr": 0.001,
        "device": "cpu",
        "log_every": 100,
    }
    model = Compressor("ae", 3, 42).fit(x, config)
    history = model.info["history"]
    assert history[-1]["corpus_probe_reconstruction"] < (
        history[0]["corpus_probe_reconstruction"] * 0.1
    )
    assert np.mean(np.sum((model.reconstruct(x) - x) ** 2, axis=1)) < 0.02
    model.save(tmp_path)
    deployed = Compressor.load(tmp_path)
    assert "decoder" not in deployed.arrays
    np.testing.assert_allclose(deployed.transform(x), model.transform(x))


@pytest.mark.parametrize("method", ["rae", "qrae", "missing"])
def test_unimplemented_methods_cannot_silently_run(method):
    with pytest.raises(ValueError, match="not implemented"):
        Compressor(method, 4)


def test_invalid_zero_embeddings_and_dimensions_fail():
    with pytest.raises(ValueError, match="Zero"):
        unit_vectors(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="entire"):
        Compressor("original", 2).fit(np.ones((4, 3)))

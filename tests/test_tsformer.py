import numpy as np
import torch

from tsformer import Standardizer, TSTransformer, make_features, make_windows, synthetic_returns, walk_forward_folds
from tsformer.train import predict, train_transformer


def test_windows_do_not_overlap_labels():
    feats = np.arange(20, dtype=np.float32).reshape(-1, 1)
    target = np.arange(20, dtype=np.float32)
    x, y = make_windows(feats, target, lookback=5, horizon=2)
    assert x.shape == (14, 5, 1)
    # first window covers t=0..4, label is t=5+t=6
    assert x[0, -1, 0] == 4 and y[0] == 5 + 6


def test_features_are_causal():
    r = synthetic_returns(500, seed=3)
    f = make_features(r)
    r2 = r.copy()
    r2[300:] *= 5
    assert np.allclose(make_features(r2)[:300], f[:300])


def test_folds_are_ordered_and_disjoint():
    for fold in walk_forward_folds(1000, n_folds=3, gap=5):
        assert fold.train.stop + 5 <= fold.val.start
        assert fold.val.stop + 5 <= fold.test.start


def test_standardizer_uses_train_only():
    tr = np.ones((10, 4, 2), dtype=np.float32)
    te = np.full((3, 4, 2), 100, dtype=np.float32)
    sc = Standardizer().fit(tr)
    assert np.allclose(sc.mean_, 1.0)


def test_model_output_shape():
    m = TSTransformer(n_features=3, lookback=16)
    assert m(torch.randn(8, 16, 3)).shape == (8,)


def test_transformer_learns_a_simple_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1200, 10, 1)).astype(np.float32)
    y = (0.8 * x[:, -1, 0] + 0.05 * rng.standard_normal(1200)).astype(np.float32)
    model = train_transformer(x[:900], y[:900], x[900:1000], y[900:1000], epochs=25, patience=8)
    p = predict(model, x[1000:])
    assert np.corrcoef(p, y[1000:])[0, 1] > 0.8

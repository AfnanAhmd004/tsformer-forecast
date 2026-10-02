"""Compare the Transformer against zero and ridge baselines with walk-forward evaluation."""
from tsformer import make_features, make_windows, synthetic_returns, walk_forward

LOOKBACK = 30

r = synthetic_returns(n=3000, seed=0)
x, y = make_windows(make_features(r), r, lookback=LOOKBACK, horizon=1)
results = walk_forward(x, y, n_folds=4, gap=1, epochs=25)
print(f"{'model':<12}{'mse':>12}{'IC':>8}{'hit':>8}")
for name, m in results.items():
    print(f"{name:<12}{m['mse']:>12.3e}{m['ic']:>8.3f}{m['hit_rate']:>8.3f}")

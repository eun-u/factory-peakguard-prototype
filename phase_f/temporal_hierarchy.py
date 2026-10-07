"""Origin-aligned temporal hierarchy for Chronos-2 (Athanasopoulos et al., EJOR 2017;
Nystrup et al., EJOR 2020).

At origin o the next 16 quarters form a hierarchy of aggregation k in {1,2,4,8,16}:
16 + 8 + 4 + 2 + 1 = 31 nodes. Level-k base forecasts come from Chronos-2 run on the
origin-aligned k-quarter mean series (bucket j ends at o - k*j; last 2048 buckets),
so only power at timestamps <= origin is read. Bottom-level forecasts are reconciled
with WLS using per-node residual variances estimated on training origins only.

Inference (Chronos env):  python -m phase_f.temporal_hierarchy --scope dev|full --out <parquet>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (1, 2, 4, 8, 16)
N_AGG = 2048
Q = pd.Timedelta(minutes=15)
NODES = [(k, s) for k in LEVELS for s in range(1, 16 // k + 1)]


def summing_matrix() -> np.ndarray:
    S = np.zeros((len(NODES), 16))
    for r, (k, s) in enumerate(NODES):
        S[r, (s - 1) * k:s * k] = 1
    return S


def aggregate_context(values: np.ndarray, i: int, k: int) -> np.ndarray:
    """Mean of k-quarter buckets ending at index i (inclusive); NaN if < k/2 observed."""
    n = min(N_AGG * k, i + 1)
    n -= n % k
    seg = values[i - n + 1:i + 1].reshape(-1, k)
    observed = np.isfinite(seg).sum(1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(seg, 1)
    mean[observed < k / 2] = np.nan
    return mean.astype("float32")


def base_matrix(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Long (origin, k, step, q50) -> origins and (n, 31) base forecasts in sum units."""
    wide = frame.pivot_table(index="origin", columns=["k", "step"], values="q50")[NODES]
    return pd.DatetimeIndex(wide.index), wide.to_numpy() * np.array([k for k, _ in NODES])


def wls_variance_projection(residuals: np.ndarray) -> np.ndarray:
    """G (16 x 31) with W = diag(node residual variance) from training residual rows."""
    S = summing_matrix()
    rows = residuals[np.isfinite(residuals).all(1)]
    if len(rows) < 100:
        raise ValueError("Too few complete training residual rows for WLS variance scaling")
    Wi = np.diag(1.0 / np.var(rows, axis=0))
    return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)


def node_actuals(power: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    bottom = np.stack([power.reindex(origins + Q * j).to_numpy(float) for j in range(1, 17)], 1)
    return bottom @ summing_matrix().T


def main(argv=None):
    import torch
    from chronos import Chronos2Pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("dev", "full"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.scope == "dev":
        from phase_c.data import load_history
        from phase_f.wf_harness import build_weekly_contexts
        history, _ = load_history(ROOT)
        contexts = build_weekly_contexts(history)
        origins = sorted(set().union(*[set(c[r]) for c in contexts.values() for r in ("fit", "stop", "cal", "score")]))
    else:
        from phase_f.final_fg_r8 import full_history
        history, _ = full_history()
        start = history.index[0] + Q * N_AGG
        origins = list(history.index[(history.index >= start) & (history.index + Q * 4 <= history.index.max())])
    values = history.power.astype("float32").to_numpy()
    pos = pd.Series(np.arange(len(values)), index=history.index)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
                                            device_map="cuda", torch_dtype=torch.float32)
    rows, t0 = [], time.time()
    for k in LEVELS:
        steps = 16 // k
        probe = origins[len(origins) // 2]
        altered = values.copy()
        altered[pos[probe] + 1:] = 10000.
        with torch.inference_mode():
            a, _ = pipe.predict_quantiles([aggregate_context(values, pos[probe], k)], prediction_length=steps, quantile_levels=[.5])
            b, _ = pipe.predict_quantiles([aggregate_context(altered, pos[probe], k)], prediction_length=steps, quantile_levels=[.5])
        if float(torch.max(torch.abs(a[0] - b[0]))) != 0:
            raise AssertionError(f"Future perturbation changed level-{k} forecasts")
        for s in range(0, len(origins), 32):
            batch = origins[s:s + 32]
            with torch.inference_mode():
                q, _ = pipe.predict_quantiles([aggregate_context(values, pos[o], k) for o in batch],
                                              prediction_length=steps, quantile_levels=[.5], batch_size=32)
            for o, qq in zip(batch, q):
                v = qq.detach().cpu().numpy()[0, :, 0]
                rows += [(o, k, j + 1, float(v[j])) for j in range(steps)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["origin", "k", "step", "q50"]).to_parquet(args.out, index=False)
    print({"origins": len(origins), "rows": len(rows), "seconds": round(time.time() - t0, 1),
           "future_perturbation": 0.0})


if __name__ == "__main__":
    main()

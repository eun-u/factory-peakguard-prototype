"""Zero-shot Chronos-2 quantile forecasts for every development origin.

Runs in the Chronos environment (outputs/phase_f/env): torch + chronos-forecasting 2.3.2.
Input context is power at timestamps <= origin only; a future-perturbation
check must return 0 before any batch is written.

Usage: python -m phase_f.gate_chronos_infer --context 2048 --out <parquet>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
LEVELS = (.05, .1, .5, .9, .95)


def main(argv=None):
    import torch
    from chronos import Chronos2Pipeline
    from phase_c.data import load_history
    from phase_f.wf_harness import build_weekly_contexts

    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    history, _ = load_history(ROOT)
    contexts = build_weekly_contexts(history)
    origins = sorted(set().union(*[set(c[r]) for c in contexts.values()
                                   for r in ("fit", "stop", "cal", "score")]))
    values = history.power.astype("float32").to_numpy()
    pos = pd.Series(np.arange(len(values)), index=history.index)

    def series(origin, x=values):
        i = pos[origin]
        return x[max(0, i - args.context + 1):i + 1].copy()

    pipe = Chronos2Pipeline.from_pretrained(MODEL, revision=REVISION, device_map="cuda",
                                            torch_dtype=torch.float32)
    probe = origins[len(origins) // 2]
    altered = values.copy()
    altered[pos[probe] + 1:] = 10000.
    with torch.inference_mode():
        a, _ = pipe.predict_quantiles([series(probe)], prediction_length=16,
                                      quantile_levels=list(LEVELS), context_length=args.context)
        b, _ = pipe.predict_quantiles([series(probe, altered)], prediction_length=16,
                                      quantile_levels=list(LEVELS), context_length=args.context)
    diff = float(torch.max(torch.abs(a[0] - b[0])))
    if diff != 0:
        raise AssertionError(f"Future perturbation changed Chronos forecast: {diff}")

    rows, t0 = [], time.time()
    for start in range(0, len(origins), args.batch):
        batch = origins[start:start + args.batch]
        with torch.inference_mode():
            q, _ = pipe.predict_quantiles([series(o) for o in batch], prediction_length=16,
                                          quantile_levels=list(LEVELS), batch_size=args.batch,
                                          context_length=args.context)
        for origin, quantiles in zip(batch, q):
            arr = quantiles.detach().cpu().numpy()[0]
            for h in range(4, 17):
                rows.append((origin, h, *arr[h - 1]))
    frame = pd.DataFrame(rows, columns=["origin", "horizon", "ch_q05", "ch_q10", "ch_q50",
                                        "ch_q90", "ch_q95"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print({"origins": len(origins), "rows": len(frame), "seconds": round(time.time() - t0, 1),
           "future_perturbation_max_abs_difference": diff})


if __name__ == "__main__":
    main()

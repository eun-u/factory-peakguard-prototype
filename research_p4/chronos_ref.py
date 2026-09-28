"""E5 (reference only): zero-shot Chronos-2 median forecasts from the sealed history."""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import torch

from .common import CACHE, HORIZONS, Timer, config, contexts, history, make_rows, safe_power, save

MODEL_ID, CONTEXT, BATCH = "amazon/chronos-2", 2048, 32


def run() -> dict:
    torch.set_num_threads(2)
    from chronos import BaseChronosPipeline
    import chronos
    cfg, df = config(), history()
    ctx = contexts(df, cfg)
    power = safe_power(df)
    grid = pd.date_range(power.index.min(), power.index.max(), freq="15min")
    series = power.reindex(grid).to_numpy(dtype=np.float32)
    pos = pd.Series(np.arange(len(grid)), index=grid)
    origins = sorted(set().union(*[set(c["cal"]) | set(c["score"]) for c in ctx.values()]))
    path = CACHE / "chronos2_paths.pkl"
    paths = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [o for o in origins if o not in paths]
    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu")
    t = Timer()
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        inputs = []
        for o in batch:
            end = int(pos[o]) + 1  # inclusive of the origin interval
            inputs.append(torch.tensor(series[max(0, end - CONTEXT):end]))
        q, _ = pipe.predict_quantiles(inputs, prediction_length=96, quantile_levels=[0.5])
        for o, item in zip(batch, q):
            paths[o] = np.asarray(item).reshape(-1, 96)[0]
        if (start // BATCH) % 25 == 0:
            print(f"[chronos] {start + len(batch)}/{len(todo)} {t()}s", flush=True)
            path.write_bytes(pickle.dumps(paths))
    path.write_bytes(pickle.dumps(paths))
    rows, meta = [], []
    for h in HORIZONS:
        for f in range(3):
            c = ctx[(h, f)]
            pick = lambda idx: np.array([paths[o][h - 1] for o in idx], dtype=float)
            rows.append(make_rows(c, h, f, "p4_chronos2_ref", pick(c["cal"]), pick(c["score"]), reference_only=True))
            meta.append({"horizon": h, "fold": f})
    save(rows, "p4_chronos2_ref", {"model": MODEL_ID, "context": CONTEXT, "chronos_version": getattr(chronos, "__version__", "2.3.2"),
                                    "torch": torch.__version__, "origins": len(origins), "seconds": t()})
    return {"origins": len(origins)}


if __name__ == "__main__":
    run()

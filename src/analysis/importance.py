"""영향변수: 특징군·개별 특징 permutation importance와 시각 × 완료 생산량 부분의존(PDP).

동결된 1시간 LightGBM을 다시 학습하지 않고, 테스트 입력의 열을 섞었을 때 오차가 얼마나 늘어나는지 본다.
중요도는 '모델이 의존하는 정도'이며 전력에 대한 인과 효과가 아니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

GROUPS = {
    "최근 전력(0~2시간)": ["current", "lag_1", "lag_2", "lag_4", "lag_8", "recent_hour_mean"],
    "전일·전주 같은 시각": ["lag_92", "lag_96", "lag_668", "lag_672", "recent_day_mean"],
    "완료 생산량": ["known_production", "known_production_prev_day"],
    "목표 시각 달력": ["target_hour", "target_quarter", "target_weekday", "target_month",
                   "target_day_of_year", "target_is_weekend"],
}


def permutation_importance(model, x: pd.DataFrame, y: pd.Series, peak: float, repeats: int = 10,
                           seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual = y.to_numpy()
    event = actual > peak

    def errors(frame):
        e = np.abs(actual - model.predict(frame))
        return e.mean(), e[event].mean()

    base_mae, base_peak = errors(x)
    units = [(name, cols) for name, cols in GROUPS.items()] + [(c, [c]) for c in x.columns]
    rows = []
    for name, cols in units:
        cols = [c for c in cols if c in x.columns]
        deltas = []
        for _ in range(repeats):
            shuffled = x.copy()
            order = rng.permutation(len(x))
            shuffled[cols] = x[cols].to_numpy()[order]
            mae, peak_mae = errors(shuffled)
            deltas.append((mae - base_mae, peak_mae - base_peak))
        deltas = np.asarray(deltas)
        rows.append({"unit": name, "kind": "group" if name in GROUPS else "feature",
                     "delta_mae": float(deltas[:, 0].mean()), "delta_mae_sd": float(deltas[:, 0].std(ddof=1)),
                     "delta_peak_mae": float(deltas[:, 1].mean()), "delta_peak_mae_sd": float(deltas[:, 1].std(ddof=1))})
    out = pd.DataFrame(rows)
    out.attrs["base_mae"], out.attrs["base_peak_mae"] = float(base_mae), float(base_peak)
    return out.sort_values(["kind", "delta_peak_mae"], ascending=[True, False])


def interaction_pdp(model, x: pd.DataFrame, features=("target_hour", "known_production")) -> pd.DataFrame:
    """두 특징의 2차원 부분의존. 생산량 축은 테스트 입력의 5~95% 분위 범위의 격자."""
    result = partial_dependence(model, x.astype(float), list(features), grid_resolution=12,
                                percentiles=(0.05, 0.95), kind="average")
    grid0, grid1 = result["grid_values"]
    values = result["average"][0]
    return pd.DataFrame(values, index=pd.Index(grid0, name=features[0]), columns=pd.Index(grid1, name=features[1]))

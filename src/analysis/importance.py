"""Permutation importance on saved validation-fold models and OOF targets."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ._common import write_table


def run_importance(history: pd.DataFrame, predictions: pd.DataFrame, cfg: dict,
                   outdir: Path, model_dir: Path | None = None,
                   selected_models: dict[int, str] | None = None) -> dict:
    if model_dir is None or not model_dir.is_dir():
        return {"status": "unsupported", "reason": "Saved validation-fold models unavailable; no feature attribution inferred from OOF predictions"}
    from src.features import build_features

    rows = []
    for path in sorted(model_dir.glob("development_h*_fold*_*.joblib")):
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            continue
        h = int(bundle["horizon"])
        fold = bundle["fold"]
        model_name = str(bundle.get("model_name", "lgbm"))
        selected = (selected_models or {}).get(h, "lgbm_no_holiday")
        # Statistical baselines have no fitted feature attribution. Explain
        # the saved development LightGBM comparator instead of that baseline.
        attributed = selected if selected.startswith("lgbm") else "lgbm_no_holiday"
        if model_name != attributed:
            continue
        role = "selected_point_model" if selected == attributed else "development_comparator_not_selected"
        score = predictions.loc[predictions.horizon.eq(h) & predictions.fold.eq(fold) & predictions.model.eq(model_name)]
        if score.empty:
            continue
        origins = pd.DatetimeIndex(score.origin)
        x, latest = build_features(history, origins, h, {**cfg, "_tau": float(bundle["tau"])})
        x.index = pd.DatetimeIndex(x.index)
        if (pd.DatetimeIndex(latest) > origins).any():
            raise AssertionError("Rebuilt importance feature uses a future observation")
        x = x.reindex(columns=bundle.get("feature_names", list(x.columns)))
        valid = x.notna().all(axis=1)
        x = x.loc[valid]
        targets = score.drop_duplicates("origin").set_index("origin").y
        common = x.index.intersection(targets.index)
        x = x.loc[common]
        y = targets.loc[common]
        if len(x) < 25:
            continue
        max_rows = int(cfg.get("analysis", {}).get("importance_max_rows", 1500))
        if len(x) > max_rows:
            take = np.linspace(0, len(x)-1, max_rows, dtype=int)
            x, y = x.iloc[take], y.iloc[take]
        result = permutation_importance(bundle["model"], x, y, scoring="neg_mean_absolute_error",
                                        n_repeats=int(cfg.get("analysis", {}).get("importance_repeats", 3)),
                                        random_state=int(cfg.get("seed", 42)), n_jobs=1)
        rows.extend({"horizon": h, "fold": fold, "model": model_name,
                     "selected_model": selected, "attribution_role": role,
                     "feature": name, "mae_increase": float(value), "repeat_std": float(sd),
                     "scored_rows": len(x)} for name, value, sd in
                    zip(x.columns, result.importances_mean, result.importances_std))
    if not rows:
        return {"status": "unsupported", "reason": "No compatible saved validation models and OOF rows"}
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["horizon", "model", "selected_model", "attribution_role", "feature"],
                             as_index=False).mae_increase.mean()
    summary = summary.sort_values(["horizon", "mae_increase"], ascending=[True, False])
    summary["rank"] = summary.groupby("horizon").cumcount()+1
    paths = {"detail": str(write_table(detail, outdir/"tables"/"permutation_importance_detail.csv")),
             "top10": str(write_table(summary.loc[summary["rank"] <= 10], outdir/"tables"/"permutation_importance_top10.csv"))}
    return {"status": "ok", "paths": paths,
            "attribution_models": detail[["horizon", "model", "selected_model", "attribution_role"]]
                                  .drop_duplicates().to_dict("records"),
            "warning": "검증 폴드 순열 중요도는 연관된 특징의 예측 기여 진단이며 인과 중요도가 아닙니다."}

"""개발 구간(시험 이전) rolling-origin 평가: 사전 검증 재현, 예측거리 곡선, conformal 보정.

시험 구간(마지막 15%)은 여기서 사용하지 않는다. 각 폴드의 피크 경계는 그 폴드의 학습 구간에서 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import block_bootstrap
from ..features import feature_frame
from ..models.baselines import naive_predict, persistence_predict
from ..models.lgbm_point import regression_model
from ..models.lgbm_quantile import conformal_offset
from ..split import rolling_origin_folds


def rolling_origin(cfg: dict, x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> list[dict]:
    """사전 검증 05_rolling_origin.csv와 같은 계산. persistence(최근 15분)를 추가로 기록한다."""
    rows = []
    n = len(x)
    ro = cfg["rolling_origin"]
    purge = cfg["one_hour"]["purge_origins"]
    for fold, (train_end, start, end) in enumerate(rolling_origin_folds(n, ro["start_fractions"], ro["width_fraction"], purge), 1):
        train_x, train_y = x.iloc[:train_end], y.iloc[:train_end]
        fold_x, fold_y, fold_meta = x.iloc[start:end], y.iloc[start:end], meta.iloc[start:end]
        if meta.iloc[train_end - 1]["target_time"] >= fold_x.index.min():
            raise AssertionError("Rolling-origin training label overlaps evaluation origin")
        model = regression_model(cfg).fit(train_x, train_y)
        peak = float(train_y.quantile(cfg["one_hour"]["peak_quantile"]))
        frame = pd.DataFrame({"y": fold_y, "date": fold_meta["date"],
                              "day": naive_predict(fold_x, "day"), "week": naive_predict(fold_x, "week"),
                              "average": naive_predict(fold_x, "average"),
                              "lgb": model.predict(fold_x),
                              "persistence": persistence_predict(fold_x, "latest_15m")})

        def metrics(f):
            actual = f["y"].to_numpy()
            event = actual > peak
            answer = {}
            for name in ("day", "week", "average", "lgb", "persistence"):
                e = np.abs(actual - f[name].to_numpy())
                answer[f"{name}_mae"] = float(np.mean(e))
                answer[f"{name}_peak_mae"] = float(np.mean(e[event])) if event.any() else float("nan")
            return answer

        point, ci = block_bootstrap(frame, metrics, cfg["bootstrap_draws"], cfg["seed"])
        for name in ("day", "week", "average", "lgb", "persistence"):
            rows.append({"fold": fold, "model": name, "training_rows": len(train_x),
                         "evaluation_rows": len(fold_x), "evaluation_first": str(fold_x.index.min()),
                         "evaluation_last": str(fold_x.index.max()), "peak_threshold": peak,
                         "peak_events": int((fold_y > peak).sum()),
                         "mae": point[f"{name}_mae"], "mae_ci_low": ci[f"{name}_mae"][0],
                         "mae_ci_high": ci[f"{name}_mae"][1],
                         "peak_mae": point[f"{name}_peak_mae"],
                         "peak_mae_ci_low": ci[f"{name}_peak_mae"][0],
                         "peak_mae_ci_high": ci[f"{name}_peak_mae"][1]})
    return rows


def _pooled_folds(cfg: dict, series: pd.DataFrame, horizon: int):
    """1시간 설정과 같은 비율의 폴드. 모든 폴드는 시험 구간(85% 지점 이후) 앞에서 끝난다."""
    x, y, meta = feature_frame(series, horizon, cfg["one_hour"]["lags"])
    ro = cfg["rolling_origin"]
    folds = rolling_origin_folds(len(x), ro["start_fractions"], ro["width_fraction"], horizon)
    dev_end = int(len(x) * cfg["one_hour"]["split_fractions"][1]) - horizon
    if any(end > dev_end for _, _, end in folds):
        raise AssertionError("Development fold reaches the held-out test period")
    for train_end, start, _ in folds:
        if meta.iloc[train_end - 1]["target_time"] >= x.index[start]:
            raise AssertionError("Fold training label overlaps evaluation origin")
    return x, y, meta, folds


def horizon_curve(cfg: dict, series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in cfg["horizon_curve"]["horizons_steps"]:
        x, y, meta, folds = _pooled_folds(cfg, series, horizon)
        frames = []
        for fold, (train_end, start, end) in enumerate(folds, 1):
            model = regression_model(cfg).fit(x.iloc[:train_end], y.iloc[:train_end])
            peak = float(y.iloc[:train_end].quantile(cfg["one_hour"]["peak_quantile"]))
            fx = x.iloc[start:end]
            frames.append(pd.DataFrame({"y": y.iloc[start:end].to_numpy(), "date": meta["date"].iloc[start:end].to_numpy(),
                                        "event": (y.iloc[start:end] > peak).to_numpy(),
                                        "persistence": persistence_predict(fx, "latest_15m"),
                                        "week": naive_predict(fx, "week", horizon),
                                        "day": naive_predict(fx, "day", horizon),
                                        "lgb": model.predict(fx)}))
        frame = pd.concat(frames, ignore_index=True)

        def metrics(f):
            out = {}
            event = f["event"].to_numpy()
            for name in ("persistence", "day", "week", "lgb"):
                e = np.abs(f["y"].to_numpy() - f[name].to_numpy())
                out[f"{name}_mae"] = float(e.mean())
                out[f"{name}_peak_mae"] = float(e[event].mean()) if event.any() else float("nan")
            best = min(out["persistence_peak_mae"], out["day_peak_mae"], out["week_peak_mae"])
            out["lgb_vs_best_baseline_peak"] = float(1 - out["lgb_peak_mae"] / best) if best > 0 else float("nan")
            return out

        point, ci = block_bootstrap(frame, metrics, cfg["bootstrap_draws"], cfg["seed"])
        for key, value in point.items():
            rows.append({"horizon_minutes": horizon * 15, "metric": key, "value": value,
                         "ci_low": ci[key][0], "ci_high": ci[key][1], "rows": len(frame),
                         "peak_positions": int(frame["event"].sum())})
    return pd.DataFrame(rows)


def conformal_dev(cfg: dict, x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    """각 개발 폴드: 학습 앞부분으로 분위수 모델 학습, 뒷부분(보정 세트)으로 δ 계산, 폴드 평가 창에서 커버리지."""
    ro = cfg["rolling_origin"]
    purge = cfg["one_hour"]["purge_origins"]
    frac = cfg["conformal"]["calibration_fraction"]
    rows, frames = [], []
    for fold, (train_end, start, end) in enumerate(rolling_origin_folds(len(x), ro["start_fractions"], ro["width_fraction"], purge), 1):
        cal_start = int(train_end * (1 - frac))
        fit_end = cal_start - purge
        fx = x.iloc[start:end]
        frame = pd.DataFrame({"y": y.iloc[start:end].to_numpy(), "date": meta["date"].iloc[start:end].to_numpy()})
        for level in cfg["conformal"]["targets"]:
            model = regression_model(cfg, alpha=level).fit(x.iloc[:fit_end], y.iloc[:fit_end])
            delta = conformal_offset(y.iloc[cal_start:train_end].to_numpy(), model.predict(x.iloc[cal_start:train_end]), level)
            pred = model.predict(fx)
            frame[f"raw_{level}"] = pred
            frame[f"conf_{level}"] = pred + delta
            rows.append({"fold": fold, "level": level, "delta": delta,
                         "calibration_rows": train_end - cal_start, "fit_rows": fit_end})
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)

    def metrics(f):
        out = {}
        for level in cfg["conformal"]["targets"]:
            for kind in ("raw", "conf"):
                cover = float(np.mean(f["y"].to_numpy() <= f[f"{kind}_{level}"].to_numpy()))
                out[f"{kind}_{level}_coverage"] = cover
                out[f"{kind}_{level}_coverage_error"] = abs(cover - level)
                out[f"{kind}_{level}_mean_bound"] = float(f[f"{kind}_{level}"].mean())
        return out

    point, ci = block_bootstrap(pooled, metrics, cfg["bootstrap_draws"], cfg["seed"])
    summary = pd.DataFrame([{"metric": k, "value": v, "ci_low": ci[k][0], "ci_high": ci[k][1]} for k, v in point.items()])
    return summary, pd.DataFrame(rows)

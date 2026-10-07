"""LightGBM 점예측·분위수 회귀 공통 설정. 사전 검증과 같은 고정 하이퍼파라미터를 쓴다."""

from __future__ import annotations

import lightgbm as lgb


def regression_model(cfg: dict, objective: str = "regression", alpha: float | None = None) -> lgb.LGBMRegressor:
    kw = dict(cfg["lightgbm"])
    kw["random_state"] = cfg["seed"]
    if alpha is not None:
        kw.update(objective="quantile", alpha=alpha)
    else:
        kw.update(objective=objective)
    return lgb.LGBMRegressor(**kw)

"""피크 초과 확률을 직접 학습하는 LightGBM 분류기 (방식 B)."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def fit_classifier(cfg: dict, x: pd.DataFrame, event: pd.Series) -> lgb.LGBMClassifier:
    p = cfg["lightgbm"]
    model = lgb.LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                               num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                               colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                               max_depth=p["max_depth"], random_state=cfg["seed"],
                               n_jobs=p["n_jobs"], verbosity=p["verbosity"])
    return model.fit(x, event)

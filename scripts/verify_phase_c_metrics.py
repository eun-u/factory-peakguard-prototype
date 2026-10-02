"""Independently recompute saved Phase C metrics from new prediction artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase_c"


def main() -> None:
    strict = json.loads((OUT / "logs/artifact_integrity_audit.json").read_text(encoding="utf-8"))
    if strict["status"] != "passed":
        raise RuntimeError("Strict row validation must pass first")
    contexts = joblib.load(OUT / "models/development_contexts.joblib")["contexts"]
    for key, context in contexts.items():
        expected_tau = np.quantile(context["y"].loc[context["fit"]].to_numpy(float), .95)
        np.testing.assert_allclose(context["tau"], expected_tau, rtol=0, atol=1e-10,
                                   err_msg=f"Fit-label Q95 differs at {key}")
    predictions = pd.read_parquet(OUT / "predictions/main_predictions.parquet")
    key_columns = ["horizon","fold","origin","target_time"]
    original_main_rows = len(predictions)
    finite = predictions.loc[np.isfinite(predictions.pred)]
    widths = finite.groupby(key_columns).model.nunique()
    common = widths.index[widths.eq(10)]
    reference = OUT / "predictions/reference_predictions.parquet"
    if reference.exists():
        predictions = pd.concat([predictions,pd.read_parquet(reference)],ignore_index=True)
    predictions = predictions.loc[pd.MultiIndex.from_frame(predictions[key_columns]).isin(common)].copy()
    assert np.isfinite(predictions.pred).all(), "Nonfinite value survived the verified paired cohort"
    predictions["absolute_error"] = (predictions.pred-predictions.y).abs()
    predictions["squared_error"] = (predictions.pred-predictions.y)**2
    predictions["actual_peak"] = predictions.y > predictions.tau
    result = {"development_only": True, "status": "passed",
              "checked_at": datetime.now(timezone.utc).isoformat(), "fit_tau_checks":len(contexts),
              "raw_main_rows": original_main_rows, "paired_keys_per_model": len(common),
              "cohort_policy": "Preregistered shared MAIN10 finite keys; structural CBL invalidity independently verified before this audit"}
    for name, group in (("model_horizon_fold_metrics",["model","horizon","fold"]),
                        ("model_horizon_pooled_metrics",["model","horizon"])):
        saved = pd.read_csv(OUT / f"tables/{name}.csv").set_index(["dataset",*group])
        seen = set()
        for dataset, frame in (("D1",predictions),("D2",predictions.loc[predictions.d2])):
            for key, cell in frame.groupby(group,sort=True):
                full_key = (dataset,*key)
                row = saved.loc[full_key]
                assert int(row.n) == len(cell), full_key
                assert int(row.peak_n) == int(cell.actual_peak.sum()), full_key
                expected = [cell.absolute_error.mean(),
                            cell.loc[cell.actual_peak,"absolute_error"].mean(),
                            np.sqrt(cell.squared_error.mean())]
                np.testing.assert_allclose(row[["MAE","Peak_MAE","RMSE"]].to_numpy(float),
                                           expected,rtol=1e-12,atol=1e-9,equal_nan=True,
                                           err_msg=str(full_key))
                seen.add(full_key)
        assert seen == set(saved.index), f"Unexpected or absent rows in {name}"
        result[name+"_rows_checked"] = len(seen)
    pooled = pd.read_csv(OUT / "tables/model_horizon_pooled_metrics.csv")
    auc = pd.read_csv(OUT / "tables/model_auc_summary.csv").set_index(["dataset","model"])
    for key, cell in pooled.groupby(["dataset","model"]):
        assert set(cell.horizon) == set(range(4,17)), key
        row = auc.loc[key]
        np.testing.assert_allclose([row.AUC_MAE,row.AUC_PeakMAE],
                                   [cell.MAE.mean(),cell.Peak_MAE.mean(skipna=False)],
                                   rtol=1e-12,atol=1e-9,equal_nan=True,err_msg=str(key))
    result["auc_rows_checked"] = len(auc)
    result["bootstrap_validation"] = "Synthetic evaluator tests and independent source review; this audit recomputes point estimates, not CI draws."
    target = OUT / "logs/metric_recomputation_audit.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()

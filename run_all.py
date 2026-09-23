"""One-command development, analysis and report pipeline with dated holdout lock."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import pandas as pd
import yaml

from src.workflow import (check_evidence, fingerprint, final_gate, json_write, audit_historical_predictions,
                          development_artifacts, verify_development_cache)

ROOT = Path(__file__).resolve().parent
STEPS = ["data", "development", "final", "analysis", "report", "package"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--only", choices=STEPS)
    parser.add_argument("--from", dest="from_step", choices=STEPS)
    parser.add_argument("--rebuild-dev", action="store_true", help="Rebuild development only; never reevaluate completed test")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg.get("output_dir", "outputs"))
    for directory in ("logs", "tables", "predictions", "figures", "models"):
        (out/directory).mkdir(parents=True, exist_ok=True)
    evidence = check_evidence(ROOT)
    run_fingerprint = fingerprint(ROOT)
    dev_fingerprint = fingerprint(ROOT, development_only=True)
    steps = [args.only] if args.only else STEPS[STEPS.index(args.from_step):] if args.from_step else STEPS
    status = {"status": "running", "steps": {}, "holdout": "locked", "fingerprint": run_fingerprint}
    started = time.perf_counter()
    data_path = Path("data/processed/power_15min.parquet")
    cache_path = out/"logs/development_cache.json"
    freeze_path = out/"logs/freeze_record.json"
    frozen_record = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    completed_freeze = frozen_record.get("status") == "completed"
    if completed_freeze:
        status["holdout"] = "frozen_evaluated"
    df = None
    development = None

    def data():
        nonlocal df
        if df is None:
            from src.data import load_power_data
            # The raw source is pinned; a stale or modified Parquet cannot change training.
            df, _ = load_power_data(cfg["source"])
        return df

    def dev_results():
        nonlocal development
        if development is None:
            if not verify_development_cache(out, dev_fingerprint):
                raise RuntimeError("Development cache is missing/stale; run python run_all.py")
            selection = json.loads((out/"logs/development_selection.json").read_text(encoding="utf-8"))
            development = {"predictions": pd.read_csv(out/"predictions/development_oof.csv", parse_dates=["origin", "target_time"]),
                           "metrics": pd.read_csv(out/"tables/development_cv.csv"), **selection}
        return development

    try:
        for step in steps:
            tick = time.perf_counter()
            print(f"[{step}] start", flush=True)
            detail = {}
            if step == "data":
                from src.data import load_power_data
                if completed_freeze:
                    from src.workflow import digest
                    if not data_path.is_file() or digest(data_path) != frozen_record["fingerprints"].get("processed_sha256"):
                        raise RuntimeError("Frozen processed artifact is missing/changed; restore the exact archived file")
                df, metadata = load_power_data(cfg["source"], None if completed_freeze else data_path)
                json_write(out/"logs/data_quality.json", metadata)
                detail = {"rows": len(df), "historical_audit": audit_historical_predictions(ROOT)}
            elif step == "development":
                cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
                if not args.rebuild_dev and verify_development_cache(out, dev_fingerprint):
                    development = dev_results()
                    detail = {"cache": "verified"}
                else:
                    if completed_freeze:
                        raise RuntimeError("Development reselection after completed freeze is forbidden; restore the verified development cache")
                    from src.training import run_development, run_t2_development
                    # Never pass holdout observations into development training, including h96 labels.
                    df_dev = data().loc[data().index < pd.Timestamp(cfg["split"]["test_start_origin"])].copy()
                    development = run_development(df_dev, cfg, out)
                    run_t2_development(df_dev, cfg, out)
                    if fingerprint(ROOT, development_only=True) != dev_fingerprint:
                        raise RuntimeError("Training code changed during execution; rebuild development before using results")
                    json_write(cache_path, {"fingerprint": dev_fingerprint, "paths": development.get("paths", {}),
                                            "artifacts": development_artifacts(out)})
                    detail = {"cache": "rebuilt", "prediction_rows": len(development["predictions"])}
            elif step == "final":
                if not final_gate(cfg):
                    detail = {"status": "date_locked", "not_before": cfg["freeze"]["not_before"],
                              "note": "No holdout evaluation performed"}
                else:
                    from src.training import freeze_and_evaluate
                    dev_results()
                    if not completed_freeze:
                        from src.data import load_power_data
                        # --from final must freeze the normalized artifact actually
                        # derived from pinned raw data, not a stale Parquet cache.
                        df, metadata = load_power_data(cfg["source"], data_path)
                        json_write(out/"logs/data_quality.json", metadata)
                    result = freeze_and_evaluate(data(), cfg, out)
                    detail = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
                    status["holdout"] = "frozen_evaluated"
            elif step == "analysis":
                from src.analysis import run_analysis
                d = dev_results()
                df_dev = data().loc[data().index < pd.Timestamp(cfg["split"]["test_start_origin"])].copy()
                first_fit_end = pd.Timestamp(d["folds"][0]["fit_end"])
                analysis = run_analysis(df_dev, d["predictions"], cfg, out,
                                        train_df=df_dev.loc[:first_fit_end], selection=d["selection"], model_dir=out/"models",
                                        tariff_2021=yaml.safe_load(Path(cfg["tariff"]["old"]).read_text(encoding="utf-8")),
                                        tariff_2026=yaml.safe_load(Path(cfg["tariff"]["current"]).read_text(encoding="utf-8")))
                json_write(out/"logs/analysis_status.json", analysis)
                detail = analysis
                freeze_record = out/"logs/freeze_record.json"
                if final_gate(cfg) and freeze_record.exists():
                    record = json.loads(freeze_record.read_text(encoding="utf-8"))
                    if record.get("status") == "completed":
                        final_analysis = run_analysis(data(), None, cfg, out,
                            train_df=df_dev.loc[:first_fit_end], selection=d["selection"],
                            tariff_2021=yaml.safe_load(Path(cfg["tariff"]["old"]).read_text(encoding="utf-8")),
                            tariff_2026=yaml.safe_load(Path(cfg["tariff"]["current"]).read_text(encoding="utf-8")),
                            scope="frozen_test", frozen_record=freeze_record)
                        json_write(out/"logs/final_analysis_status.json", final_analysis)
                        detail = {"development": analysis, "frozen_test": final_analysis}
            elif step == "report":
                from src.reporting import generate_reports
                d = dev_results()
                detail = generate_reports(data(), d, cfg, out)
            elif step == "package":
                from src.packaging import build_package
                detail = build_package(ROOT, cfg)
            status["steps"][step] = {"seconds": round(time.perf_counter()-tick, 3), "detail": detail}
            print(f"[{step}] done {status['steps'][step]['seconds']:.1f}s", flush=True)
            json_write(out/"logs/run_status.json", status)
        if check_evidence(ROOT) != evidence:
            raise AssertionError("Evidence changed during run")
        status["status"] = "completed_development" if status["holdout"] == "locked" else "completed_frozen"
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        status["seconds"] = round(time.perf_counter()-started, 3)
        from src.workflow import record_run_evidence
        record_run_evidence(ROOT, out, status)


if __name__ == "__main__":
    main()

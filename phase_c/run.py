"""Resumable Phase C entry point. Never calls legacy final-evaluation entry points."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase_c"
PREREG = OUT / "logs/preregistration_phase_c.md"
CONFIG = ROOT / "configs/phase_c.json"
PROBE = ROOT / "scripts/phase_b_reference/run_phase_b_temp_probe.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    os.replace(temporary,path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_hashes() -> dict:
    """Hash opaque bytes of the untouched baseline; do not parse artifacts."""
    raw = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT)
    return {name:sha(ROOT/name) for name in raw.decode("utf-8").split("\0") if name and (ROOT/name).is_file()}


def implementation_hashes() -> dict:
    files = [p for p in (ROOT/"phase_c").glob("*.py")]
    files += [p for p in (ROOT/"src").rglob("*.py")]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}


def protected_hashes() -> dict:
    hashes=tracked_hashes()
    for directory in ("outputs","verification","report","slides","submission","data/raw"):
        for path in (ROOT/directory).rglob("*"):
            if path.is_file() and not path.is_relative_to(OUT):
                hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    return hashes


def assert_seal() -> dict:
    lock = read_json(OUT/"logs/preregistration_lock.json")
    for relative,digest in lock["bound_inputs"].items():
        if sha(ROOT/relative) != digest:
            raise AssertionError(f"Locked input changed: {relative}")
    # Training and data semantics remain immutable; report/evaluator patches
    # need an explicit repair log and validation before rescoring.
    for relative,digest in lock["implementation_hashes"].items():
        if sha(ROOT/relative) != digest:
            raise AssertionError(f"Locked implementation changed: {relative}")
    return lock


def check_protected() -> dict:
    before = read_json(OUT/"logs/protected_files_before.json")
    after = protected_hashes()
    changed = sorted(name for name,digest in before.items() if after.get(name)!=digest)
    if changed:
        raise AssertionError(f"Historical tracked artifacts changed: {changed}")
    forbidden = [p.relative_to(ROOT).as_posix() for p in OUT.rglob("*")
                 if p.is_file() and (p.name=="final_test.csv" or "freeze_approval" in p.name)]
    if forbidden:
        raise AssertionError(f"Forbidden final artifact: {forbidden}")
    return {"protected_baseline_files_including_ignored":len(before),"modified_baseline_files":changed,
            "new_final_test_artifacts":forbidden,"preserved":True}


def prepare() -> None:
    from .data import load_history,build_contexts,CORE_FEATURES
    from .chronos_reference import cache_equivalence
    if (OUT/"logs/preregistration_lock.json").exists():
        assert_seal()
        print("Existing preregistration seal verified; preparation reused",flush=True)
        return
    cfg=read_json(CONFIG)
    required={"data.py","training.py","tcn.py","statistical.py","evaluation.py","run.py","chronos_reference.py"}
    if not required <= {p.name for p in (ROOT/"phase_c").glob("*.py")}:
        raise RuntimeError("Implementation incomplete; cannot seal")
    for folder in ("logs","models","predictions","figures","tables"):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()!="Phase3_experiments":
        raise AssertionError("Wrong working branch")
    protection_path=OUT/"logs/protected_files_before.json"
    if not protection_path.exists():
        write_json(protection_path,protected_hashes())
    else:
        check_protected()
    history,audit=load_history(ROOT)
    if audit["raw_sha256"] not in (ROOT/"data/README.md").read_text(encoding="utf-8"):
        raise AssertionError("Raw hash is not registered in data/README.md")
    contexts=build_contexts(history)
    if list(CORE_FEATURES)!=sum(cfg["feature_groups"].values(),[]):
        raise AssertionError("Feature implementation differs from preregistration")
    for c in contexts.values():
        assert not c["cal"].intersection(c["fit"].append(c["stop"]).append(c["score"])).size
        for name,times in c["provenance"].items():
            if (pd.DatetimeIndex(times)>times.index).any():
                raise AssertionError(f"Future feature {name}")
    joblib.dump({"history":history,"contexts":contexts,"audit":audit},OUT/"models/development_contexts.joblib",compress=1)
    pd.DataFrame([c["summary"]|{"tau":c["tau"]} for c in contexts.values()]).to_csv(OUT/"tables/fold_manifest.csv",index=False)
    write_json(OUT/"logs/data_preparation.json",audit)
    write_json(OUT/"logs/chronos_cache_equivalence.json",cache_equivalence(ROOT,contexts))
    import torch
    packages={name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","scikit-learn","lightgbm","torch","statsforecast","statsmodels","chronos-forecasting","transformers","pytest"]}
    environment={"development_only":True,"prepared_at":now(),"python":sys.version,
        "executable":sys.executable,"platform":platform.platform(),"packages":packages,
        "cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed":42,"branch":"Phase3_experiments",
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    write_json(OUT/"logs/run_environment.json",environment)
    (OUT/"logs/requirements_phase_c_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True) if importlib.util.find_spec("pip") else "\n".join(f"{k}=={v}" for k,v in sorted(packages.items())),encoding="utf-8")
    inputs=[CONFIG,PREREG,PROBE,ROOT/"configs/default.yaml",ROOT/"data/raw/task05_power/okm_augumented_2021.csv"]
    lock={"development_only":True,"registered_at":now(),"training_started":False,
          "bound_inputs":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
          "implementation_hashes":implementation_hashes(),
          "context_cache_sha256":sha(OUT/"models/development_contexts.joblib"),
          "historical_artifact_incident_disclosed":True}
    write_json(OUT/"logs/preregistration_lock.json",lock)
    write_json(OUT/"logs/preparation_status.json",{"development_only":True,"status":"preregistered_ready",
        "feature_count":15,"contexts":39,"training_started":False,"score_evaluation_started":False,
        "raw_holdout_observations_parsed":False,"historical_final_artifact_read":True})
    print("Preregistered: 39 contexts, 15 exact Phase B features, sealed history",flush=True)


def load_contexts():
    seal=assert_seal()
    path=OUT/"models/development_contexts.joblib"
    if sha(path)!=seal["context_cache_sha256"]:
        raise AssertionError("Development context cache changed")
    return joblib.load(path)


def finish_lock() -> dict:
    lock={"development_only":True,"locked_before_score":True,"created_at":now(),
          "preregistration_sha256":sha(PREREG),"config_sha256":sha(CONFIG)}
    for path,key in (("baseline_config_lock.json",None),("lgbm_config_lock.json","lgbm"),("tcn_config_lock.json","tcn")):
        fragment=read_json(OUT/"logs"/path)
        if key is None:
            lock.update(fragment)
        else:
            lock[key]=fragment.get(key,fragment)
    write_json(OUT/"logs/model_config_lock.json",lock)
    return lock


def qa(predictions: pd.DataFrame|None=None) -> dict:
    seal=assert_seal()
    saved=load_contexts(); contexts=saved["contexts"]
    cfg=read_json(CONFIG)
    checks={"A_holdout_timestamps_absent":True,"B_cal_selection_absent":True,
            "C_feature_provenance_causal":True,"D_purged_targets":True,
            "E_exact_13_horizons":set(contexts)=={(h,f) for h in range(4,17) for f in range(3)},
            "F_paired_timestamps":None,"G_D2_score_subset_preserves_history":True,
            "H_residual_same_configuration":None,"I_weighted_same_configuration_weight2":None,
            "J_Chronos_selection_excluded":None,"K_final_artifacts_preserved":True}
    boundary=pd.Timestamp(cfg["boundary"])
    for (h,f),c in contexts.items():
        for role in ("fit","stop","cal","score"):
            checks["A_holdout_timestamps_absent"] &= bool((c[role]<boundary).all() and (c["target_time"].loc[c[role]]<boundary).all())
        checks["B_cal_selection_absent"] &= not len(c["cal"].intersection(c["fit"].append(c["stop"]).append(c["score"])))
        checks["D_purged_targets"] &= all(c["target_time"].loc[c[a]].max()<c[b].min() for a,b in zip(("fit","stop","cal"),("stop","cal","score")))
        checks["C_feature_provenance_causal"] &= all(not (pd.DatetimeIndex(ts)>ts.index).any() for ts in c["provenance"].values())
        checks["G_D2_score_subset_preserves_history"] &= set(c["d2"].index[c["d2"]])<=set(c["score"])
    preserved=check_protected()
    if predictions is not None:
        checks["A_holdout_timestamps_absent"] &= bool((predictions.origin<boundary).all() and (predictions.target_time<boundary).all())
        for (h,f),c in contexts.items():
            got=predictions[(predictions.horizon==h)&(predictions.fold==f)]
            checks["B_cal_selection_absent"] &= not got.origin.isin(c["cal"]).any()
            if any(set(g.origin)!=set(c["score"]) for m,g in got.groupby("model")):
                raise AssertionError("Predictions are not on original score keys")
        from .evaluation import _prepare,MAIN
        paired,_=_prepare(predictions)
        checks["F_paired_timestamps"]=True
        records=read_json(OUT/"logs/model_config_lock.json")
        checks["H_residual_same_configuration"]=True
        checks["I_weighted_same_configuration_weight2"]=True
        # Inspect effective parameters persisted for every model, not merely
        # a global assertion. Training metadata supplies exact source config.
        for (h,f),g in predictions.groupby(["horizon","fold"]):
            configs={m:json.loads(rows.selected_config.iloc[0]) for m,rows in g.groupby("model")}
            if any("model_params" not in configs[m] for m in ("M1","M1-W","C1")):
                raise AssertionError("Effective LightGBM configuration evidence missing")
            for name in ("M1-W","C1"):
                if configs[name]["model_params"] != configs["M1"]["model_params"]:
                    raise AssertionError("LightGBM family configuration mismatch")
            if configs["M1-W"].get("peak_weight")!=2.0:
                raise AssertionError("Peak weight is not exactly 2")
        checks["J_Chronos_selection_excluded"]="R1" not in MAIN
    incident=read_json(OUT/"logs/boundary_incident.json")
    result={"development_only":True,"checked_at":now(),"checks":checks,
        "raw_holdout_observations_parsed":False,"calibration_labels_used_for_selection":False,
        "historical_final_artifact_read":True,"historical_artifact_incident":incident,
        "protected_files":preserved,"preregistration_sealed":True,
        "all_applicable_checks_pass":all(v is None or bool(v) for v in checks.values()),
        "cal_use_evidence":"Partition/origin assertions plus synthetic training-routing tests and source-reviewed fit/stop-only array selection; not inferred from disjointness alone"}
    write_json(OUT/"logs/leakage_audit.json",result)
    if not result["all_applicable_checks_pass"]:
        raise AssertionError("Leakage/QA check failed")
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["prepare","baselines","lgbm_tune","tcn_tune","score","download_reference","chronos","evaluate","qa","all"],default="all")
    args=parser.parse_args()
    started=perf_counter()
    runtime_path=OUT/"logs/runtime.json"
    runtime=read_json(runtime_path) if runtime_path.exists() else {"development_only":True,"first_job_started_at":now(),"stages":[]}
    stage={"name":args.stage,"started_at":now(),"status":"running"}
    runtime["stages"].append(stage); write_json(runtime_path,runtime)
    try:
        if args.stage=="download_reference":
            from .chronos_reference import download_model
            download_model(OUT,read_json(CONFIG))
        else:
            prepare()
            if args.stage!="prepare":
                saved=load_contexts(); history=saved["history"]; contexts=saved["contexts"];cfg=read_json(CONFIG)
                from .training import prepare_baselines,tune_family,score_main
                if args.stage in ("baselines","all"):
                    fragment=prepare_baselines(history,contexts,cfg,OUT)
                    write_json(OUT/"logs/baseline_config_lock.json",fragment)
                if args.stage in ("lgbm_tune","all"):
                    tune_family("lgbm",history,contexts,cfg,OUT)
                if args.stage in ("tcn_tune","all"):
                    tune_family("tcn",history,contexts,cfg,OUT)
                if args.stage in ("score","all"):
                    lock=finish_lock()
                    predictions=score_main(history,contexts,cfg,lock,OUT)
                    predictions.to_parquet(OUT/"predictions/main_predictions.parquet",index=False)
                if args.stage in ("chronos","all"):
                    from .chronos_reference import download_model,run_reference
                    if not (OUT/"logs/chronos_model_download.json").exists():
                        download_model(OUT,cfg)
                    run_reference(history,contexts,cfg,OUT)
                if args.stage in ("evaluate","qa","all"):
                    predictions=pd.read_parquet(OUT/"predictions/main_predictions.parquet")
                    ref_path=OUT/"predictions/reference_predictions.parquet"
                    if ref_path.exists():
                        predictions=pd.concat([predictions,pd.read_parquet(ref_path)],ignore_index=True)
                    qa(predictions)
                    if args.stage!="qa":
                        from .evaluation import evaluate
                        selection=evaluate(predictions,OUT,bootstrap_n=cfg["bootstrap_n"],seed=cfg["seed"])
                        write_json(OUT/"logs/model_selection.json",selection)
                elif args.stage in ("prepare","baselines"):
                    qa()
        stage["status"]="completed"
    except Exception as exc:
        stage.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        failure=OUT/"logs"/f"failure_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        failure.write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        stage["completed_at"]=now();stage["seconds"]=perf_counter()-started
        runtime["actual_job_seconds"]=sum(s.get("seconds",0) for s in runtime["stages"])
        runtime["last_updated_at"]=now();write_json(runtime_path,runtime)
        print(f"{args.stage}: {stage['status']} ({stage['seconds']:.1f}s)",flush=True)


if __name__=="__main__":
    main()

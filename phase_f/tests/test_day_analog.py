"""Causal FIT-only analog retrieval, fallback, and physical checkpoint tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import joblib

from phase_f.models import day_analog as analog
from phase_f.registry import config_hash, sha256


ROOT = Path(__file__).resolve().parents[2]


def _view(tmp_path, *, length=16, top_k=3):
    idx = pd.date_range("2021-01-01 00:15", periods=1200, freq="15min")
    slot = idx.hour.to_numpy() * 4 + idx.minute.to_numpy() // 15
    day = np.arange(len(idx)) // 96
    power = 100 + 16 * np.sin(slot * 2 * np.pi / 96) + (day % 4) * 3
    history = pd.DataFrame({"power": power, "quality_bad": False,
                            "time_repaired": False}, index=idx)
    roles = {"fit": idx[96:768], "stop": idx[800:880],
             "cal": idx[900:910], "score": idx[950:960]}
    contexts = {}
    for h in analog.HORIZONS:
        y = pd.Series(power, index=idx).shift(-h)
        contexts[(h, 0)] = {**roles, "y": y, "tau": 100.0,
                            "x": pd.DataFrame({"slot7d": 100.0}, index=idx),
                            "d2": pd.Series(False, index=idx),
                            "target_time": pd.Series(idx + h * pd.Timedelta(minutes=15),
                                                     index=idx),
                            "summary": {"arm": "EXPLORE"}}
    paths = pd.concat([pd.DataFrame({"origin": origins,
                                     "horizon": h, "r1": 100., "q50": 100.,
                                     "q10": 90., "q90": 110., "q95": 115.})
                       for h in analog.HORIZONS for origins in roles.values()],
                      ignore_index=True)
    proof = {"leakage_test": "passed", "input_cutoff_rule": "history.index <= origin",
             "future_perturbation_max_abs_difference": 0.0}
    paths.attrs["causal_provenance"] = proof
    spec = {"id": f"FG-R4-test-p{length}-k{top_k}", "adapter": "day_analog",
            "arm": "EXPLORE", "prefix_length": length, "top_k": top_k,
            "bank_days": 56, "rolling_audit_sha256": config_hash(proof)}
    view = SimpleNamespace(root=ROOT, out=tmp_path, history=history,
                           contexts=contexts, split_lock={"lock_sha256": "split"},
                           seal={"raw_sha256": "raw"})
    return view, spec, paths


def test_centered_shape_ties_recent_first_and_midnight_slot():
    idx = pd.date_range("2021-01-01 00:00", periods=6 * 96 + 1, freq="15min")
    history = pd.DataFrame({"power": 100., "quality_bad": False}, index=idx)
    origins = pd.DatetimeIndex([pd.Timestamp(f"2021-01-0{day} 00:00")
                                for day in (2, 3, 4)])
    prefix = analog._prefixes(history, origins, 16)
    assert analog._slots(origins).tolist() == [0, 0, 0]
    bank = {"origins_ns": origins.asi8, "slots": analog._slots(origins),
            "prefixes": prefix, "levels": prefix[:, -1],
            "suffixes": np.tile(np.arange(3.)[:, None], (1, 13))}
    query = pd.DatetimeIndex([pd.Timestamp("2021-01-05 00:00")])
    ranked = analog._rank(history, bank, query, 16, 3, role="score")
    assert ranked["indices"].tolist() == [[2, 1, 0]]
    assert ranked["support"].tolist() == [3]
    assert analog._analog_points(bank, ranked, 4).tolist() == [1.0]


def test_weighted_median_level_shift_and_insufficient_support():
    bank = {"suffixes": np.tile(np.array([10., 20., 100.])[:, None], (1, 13)),
            "levels": np.array([10., 10., 10.])}
    rank = {"indices": np.array([[0, 1, 2], [-1, -1, -1]]),
            "distances": np.array([[1., 2., 100.], [np.nan] * 3]),
            "query_levels": np.array([15., np.nan])}
    points = analog._analog_points(bank, rank, 4)
    assert points[0] == 15.
    assert np.isnan(points[1])
    np.testing.assert_array_equal(analog._blend(np.array([90., 91.]), points, .5),
                                  [52.5, 91.])
    with pytest.raises(ValueError, match="observed peak"):
        analog._choose_alpha(np.array([100.]), np.array([90.]), np.array([95.]),
                             np.array([False]))


def test_quality_flags_and_missing_lookups_force_r1_fallback():
    idx = pd.date_range("2021-01-01", periods=5 * 96 + 1, freq="15min")
    history = pd.DataFrame({"power": 100., "quality_bad": False,
                            "time_repaired": False}, index=idx)
    bank_origins = pd.DatetimeIndex([idx[96], idx[192], idx[288]])
    bank_prefix = analog._prefixes(history, bank_origins, 16)
    bank = {"origins_ns": bank_origins.asi8, "slots": analog._slots(bank_origins),
            "prefixes": bank_prefix, "levels": bank_prefix[:, -1],
            "suffixes": np.full((3, 13), 120.)}
    query = pd.DatetimeIndex([idx[384]])
    history.loc[idx[383], "time_repaired"] = True
    ranked = analog._rank(history, bank, query, 16, 3, role="score")
    assert not ranked["query_prefix_clean"].item()
    assert np.isnan(analog._analog_points(bank, ranked, 4)).all()
    assert analog._blend(np.array([88.]), analog._analog_points(bank, ranked, 4), 1.).item() == 88.
    short_bank = {name: value[:2] for name, value in bank.items()}
    clean = analog._rank(history.assign(time_repaired=False), short_bank,
                         query, 16, 3, role="score")
    assert clean["support"].tolist() == [2]
    assert clean["indices"].tolist() == [[-1, -1, -1]]
    history.loc[idx[383], "time_repaired"] = False
    history.loc[bank_origins[0], "quality_bad"] = True
    prefixes = analog._prefixes(history, bank_origins, 16)
    assert np.isnan(prefixes[0, -1])


def test_zero_clean_bank_cases_still_yields_audited_r1_fallback(tmp_path):
    view, spec, paths = _view(tmp_path)
    view.history.loc[:, "quality_bad"] = True
    frame, audit = analog.run(view, spec, paths, smoke_first_fold=True)
    assert audit["cells"][0]["bank_audit"]["bank_cases"] == 0
    assert frame.analog_fallback.all()
    np.testing.assert_array_equal(frame.pred, frame.r1)
    analog.verify_cells(view, spec, paths, audit, smoke_first_fold=True)


def test_bank_excludes_bad_fit_prefix_and_stop_truth_binds_identity(tmp_path):
    view, spec, paths = _view(tmp_path)
    fit, stop, _ = analog._common_roles(view.contexts, 0)
    baseline_bank, baseline_audit = analog._build_bank(view, 0, fit, 16)
    _, paths_sha = analog._paths_index(paths, view.contexts)
    identity = analog._identity(view, spec, paths_sha, 0, fit, stop,
                                baseline_bank, baseline_audit)
    altered = copy.copy(view)
    altered.contexts = copy.deepcopy(view.contexts)
    altered.contexts[(4, 0)]["y"].loc[stop[0]] += 1.
    assert analog._identity(altered, spec, paths_sha, 0, fit, stop,
                            baseline_bank, baseline_audit) != identity
    flagged = copy.copy(view)
    flagged.history = view.history.copy()
    flagged.history.loc[fit[-1], "quality_bad"] = True
    changed_bank, changed_audit = analog._build_bank(flagged, 0, fit, 16)
    assert changed_audit["bank_cases"] < baseline_audit["bank_cases"]
    assert fit[-1].value not in changed_bank["origins_ns"]
    assert analog._identity(flagged, spec, paths_sha, 0, fit, stop,
                            changed_bank, changed_audit) != identity
    target_flagged = copy.copy(view)
    target_flagged.history = view.history.copy()
    target_time = view.contexts[(16, 0)]["target_time"].loc[fit[-1]]
    target_flagged.history.loc[target_time, "quality_bad"] = True
    target_bank, target_audit = analog._build_bank(target_flagged, 0, fit, 16)
    assert target_audit["excluded_bad_or_missing_target_observation"] > 0
    assert fit[-1].value not in target_bank["origins_ns"]
    stop_changed = copy.copy(view)
    stop_changed.history = view.history.copy()
    stop_changed.history.loc[stop[0], "power"] += 5.
    assert analog._identity(stop_changed, spec, paths_sha, 0, fit, stop,
                            baseline_bank, baseline_audit) != identity


@pytest.mark.parametrize("tamper", ["alpha", "spec"])
def test_rejects_consistently_rehashed_payload_and_metadata_tampering(tmp_path, tamper):
    view, spec, paths = _view(tmp_path)
    _, audit = analog.run(view, spec, paths, smoke_first_fold=True)
    model_path, metadata_path = analog._checkpoint_paths(view, spec, 0)
    bundle = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tamper == "alpha":
        alternate = 1.0 if bundle["alpha"] != 1.0 else 0.0
        bundle["alpha"] = metadata["alpha"] = alternate
    else:
        bundle["spec"]["top_k"] = metadata["spec"]["top_k"] = 5
    joblib.dump(bundle, model_path)
    metadata["model_sha256"] = sha256(model_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    tampered_model_sha, tampered_meta_sha = sha256(model_path), sha256(metadata_path)
    with pytest.raises(RuntimeError, match="preserved physical evidence"):
        analog.verify_cells(view, spec, paths, audit, smoke_first_fold=True)
    assert sha256(next(model_path.parent.glob("f0.orphan-*.joblib"))) == tampered_model_sha
    assert sha256(next(model_path.parent.glob("f0.orphan-*.json"))) == tampered_meta_sha


@pytest.mark.parametrize("field", ["bank_audit", "stop_analog_coverage",
                                    "fit_library_frozen_through_score"])
def test_rejects_drifted_metadata_claims_with_unchanged_model(tmp_path, field):
    view, spec, paths = _view(tmp_path)
    _, audit = analog.run(view, spec, paths, smoke_first_fold=True)
    model_path, metadata_path = analog._checkpoint_paths(view, spec, 0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if field == "bank_audit":
        metadata[field]["bank_cases"] += 1
    elif field == "stop_analog_coverage":
        metadata[field] = .123
    else:
        metadata[field] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    model_sha, meta_sha = sha256(model_path), sha256(metadata_path)
    with pytest.raises(RuntimeError, match="preserved physical evidence"):
        analog.verify_cells(view, spec, paths, audit, smoke_first_fold=True)
    assert sha256(next(model_path.parent.glob("f0.orphan-*.joblib"))) == model_sha
    assert sha256(next(model_path.parent.glob("f0.orphan-*.json"))) == meta_sha


def test_fit_queries_only_use_matured_prior_targets():
    idx = pd.date_range("2021-01-01 00:00", periods=8 * 96 + 1, freq="15min")
    history = pd.DataFrame({"power": 100.}, index=idx)
    origins = pd.DatetimeIndex([idx[96], idx[192], idx[288], idx[384]])
    prefix = analog._prefixes(history, origins, 16)
    bank = {"origins_ns": origins.asi8, "slots": analog._slots(origins),
            "prefixes": prefix, "levels": prefix[:, -1],
            "suffixes": np.full((4, 13), 110.)}
    ranked = analog._rank(history, bank, origins, 16, 3, role="fit", horizon=16)
    assert ranked["support"].tolist() == [0, 1, 2, 3]
    assert ranked["indices"][2].tolist() == [-1, -1, -1]
    assert ranked["indices"][3].tolist() == [2, 1, 0]
    assert 3 not in ranked["indices"][3]


@pytest.mark.parametrize("length,top_k", [(16, 3), (96, 5)])
def test_run_preserves_keys_uses_fit_only_and_verifies_physical_checkpoint(tmp_path, length, top_k):
    view, spec, paths = _view(tmp_path, length=length, top_k=top_k)
    frame, audit = analog.run(view, spec, paths, smoke_first_fold=True)
    assert len(frame) == 13 * (10 + 10)
    assert not frame.duplicated(["fold", "horizon", "role", "origin"]).any()
    assert frame.analog_fallback.eq(frame.analog_point.isna()).all()
    assert np.isfinite(frame.pred).all()
    assert audit["n_stochastic_seeds"] == 0
    assert audit["cells"][0]["future_perturbation_max_abs_difference"] == 0.0
    assert audit["cells"][0]["bank_audit"]["same_slot_case_deduplication"] is False
    analog.verify_cells(view, spec, paths, audit, smoke_first_fold=True)
    cached, cached_audit = analog.run(view, spec, paths, smoke_first_fold=True)
    pd.testing.assert_frame_equal(frame, cached)
    assert cached_audit["cells"][0]["cache_reused"] is True

    # CAL/SCORE outcomes may be read by make_frame for later evaluation, but
    # changing them cannot change model identity, STOP alpha, or predictions.
    changed = copy.copy(view)
    changed.contexts = copy.deepcopy(view.contexts)
    for context in changed.contexts.values():
        context["y"].loc[context["cal"]] = 999.
        context["y"].loc[context["score"]] = 999.
    altered, altered_audit = analog.run(changed, spec, paths, smoke_first_fold=True)
    np.testing.assert_array_equal(frame.pred, altered.pred)
    assert audit["cells"][0]["identity"] == altered_audit["cells"][0]["identity"]


def test_rejects_corrupt_metadata_without_refit_and_preserves_bytes(tmp_path):
    view, spec, paths = _view(tmp_path)
    _, audit = analog.run(view, spec, paths, smoke_first_fold=True)
    model_path, metadata_path = analog._checkpoint_paths(view, spec, 0)
    original_model_sha = sha256(model_path)
    metadata_path.write_bytes(b"{truncated-json")
    corrupt_sha = sha256(metadata_path)
    with pytest.raises(RuntimeError, match="preserved physical evidence"):
        analog.verify_cells(view, spec, paths, audit, smoke_first_fold=True)
    assert not model_path.exists() and not metadata_path.exists()
    preserved_models = list(model_path.parent.glob("f0.orphan-*.joblib"))
    preserved_meta = list(model_path.parent.glob("f0.orphan-*.json"))
    assert len(preserved_models) == len(preserved_meta) == 1
    assert sha256(preserved_models[0]) == original_model_sha
    assert sha256(preserved_meta[0]) == corrupt_sha

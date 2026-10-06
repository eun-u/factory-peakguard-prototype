"""Walk-forward seed evidence, arm isolation, and candidate cache integrity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase_f.harness import make_frame
from phase_f import wf_models
from phase_f.registry import sha256


class WeeklyFixture:
    def __init__(self, root: Path, out: Path):
        self.root, self.out = root, out
        self.seal = {"raw_sha256": "synthetic-sealed-input"}
        self.split_lock = {"lock_sha256": "synthetic-week-lock"}
        index = pd.date_range("2021-03-01", "2021-04-25", freq="15min")
        self.history = pd.DataFrame({"power": np.arange(len(index), dtype=float) / 100 + 20}, index=index)
        self.contexts = {}
        for fold, start in ((0, "2021-04-12"), (1, "2021-04-19")):
            score = pd.date_range(start, periods=4, freq="15min")
            cal = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=2), periods=4, freq="15min")
            fit = pd.date_range("2021-03-09", periods=100, freq="15min")
            stop = pd.date_range("2021-04-01", periods=4, freq="15min")
            all_origins = fit.union(stop).union(cal).union(score)
            self.contexts[(4, fold)] = {
                "fit": fit, "stop": stop, "cal": cal, "score": score,
                "x": pd.DataFrame({"slot7d": np.full(len(all_origins), 20.)}, index=all_origins),
                "y": pd.Series(np.full(len(all_origins), 25.), index=all_origins),
                "target_time": pd.Series(all_origins + pd.Timedelta(hours=1), index=all_origins),
                "tau": 30., "d2": pd.Series(False, index=all_origins),
            }

    def origins(self, h, fold, role):
        return self.contexts[(h, fold)][role]


def _fake_adapter(view, spec):
    frames = []
    for (h, fold), context in view.contexts.items():
        for role in ("cal", "score"):
            origins = view.origins(h, fold, role)
            prediction = np.full(len(origins), 21. + spec["seed"] % 5)
            frames.append(make_frame(context, origins, h, fold, prediction, spec["id"], role))
    return pd.concat(frames, ignore_index=True), {
        "leakage_test": "passed", "cells": [{"stop_MAE": float(spec["seed"] % 5)}]}


def test_five_seed_mean_is_not_best_seed_and_reuses_only_sealed_cache(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    calls = []

    def execute(view, spec):
        calls.append(spec["seed"])
        return _fake_adapter(view, spec)

    monkeypatch.setattr(wf_models, "_execute_one", execute)
    spec = {"id": "F5-wf-test", "adapter": "neural", "kind": "dlinear"}
    mean, audit = wf_models.run(prepared, spec)
    assert calls == list(wf_models.SEEDS[:5])
    assert audit["n_seeds"] == 5 and audit["stop_cells"] == 5
    assert mean.model.eq("F5-wf-test").all()
    assert mean.arm.eq("EXPLORE").any() and not mean.arm.eq("CONFIRM").any()
    assert np.allclose(mean.pred, np.mean([21 + seed % 5 for seed in wf_models.SEEDS[:5]]))
    assert mean.loc[mean.role.eq("cal"), "cal_provenance"].eq("fit_only").all()
    seed_files = list((tmp_path / "predictions/seeds/EXPLORE/F5-wf-test").glob("*.parquet"))
    assert len(seed_files) == 5
    again, second_audit = wf_models.run(prepared, spec)
    assert calls == list(wf_models.SEEDS[:5])
    assert np.array_equal(mean.pred.to_numpy(), again.pred.to_numpy())
    assert all(row["cache_reused"] for row in second_audit["seeds"])


def test_seed_cache_tamper_fails_closed(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    monkeypatch.setattr(wf_models, "_execute_one", _fake_adapter)
    spec = {"id": "F5-wf-integrity", "adapter": "neural", "kind": "dlinear"}
    wf_models.run(prepared, spec)
    path = wf_models._seed_path(prepared, spec["id"], "EXPLORE", wf_models.SEEDS[0])
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="cache changed"):
        wf_models.run(prepared, spec)


@pytest.mark.parametrize("orphan_side", ("parquet", "manifest"))
def test_partial_seed_transaction_preserves_orphan_then_refits(tmp_path, monkeypatch, orphan_side):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    monkeypatch.setattr(wf_models, "_execute_one", _fake_adapter)
    spec = {"id": "F5-wf-orphan", "adapter": "neural", "kind": "dlinear"}
    wf_models.run(prepared, spec)
    path = wf_models._seed_path(prepared, spec["id"], "EXPLORE", wf_models.SEEDS[0])
    meta = path.with_suffix(".json")
    preserved = path if orphan_side == "parquet" else meta
    missing = meta if orphan_side == "parquet" else path
    original_sha = sha256(preserved)
    missing.unlink()  # Deliberately simulate one side of an interrupted test transaction.
    frame, audit = wf_models.run(prepared, spec)
    assert len(frame) > 0 and audit["seeds"][0]["cache_reused"] is False
    assert path.exists() and meta.exists()
    orphans = list(preserved.parent.glob(f"{preserved.stem}.orphan-{original_sha[:12]}*{preserved.suffix}"))
    assert len(orphans) == 1
    assert sha256(orphans[0]) == original_sha


@pytest.mark.parametrize("column", ("y", "tau", "d2", "fit_mean", "mase_scale"))
def test_seed_frame_rejects_changed_locked_context_values(tmp_path, column):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    view = wf_models._arm_view(prepared, "EXPLORE")
    frame, _ = _fake_adapter(view, {"id": "tampered", "seed": 42})
    index = frame.role.eq("score").idxmax()
    frame.loc[index, column] = (not bool(frame.loc[index, column])
                                if column == "d2" else float(frame.loc[index, column]) + 1)
    with pytest.raises(ValueError, match="differs from locked context|differs from fit-only context"):
        wf_models._validate_frame(view, frame, "tampered", "EXPLORE")


def test_confirm_arm_only_and_explicit_finalist_ten(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    monkeypatch.setattr(wf_models, "_execute_one", _fake_adapter)
    spec = {"id": "F5-wf-finalist", "adapter": "neural", "kind": "dlinear"}
    frame, audit = wf_models.run(prepared, spec, arm="CONFIRM", n_seeds=10)
    assert audit["n_seeds"] == 10
    assert frame.loc[frame.role.eq("score"), "arm"].eq("CONFIRM").all()
    assert len(frame) == 8  # Four cal and four score rows for the one odd week.
    assert not (tmp_path / "predictions/seeds/EXPLORE").exists()


def test_seed_contract_for_stochastic_and_fixed_families():
    assert len(wf_models._seed_list({"id": "n", "adapter": "neural"})) == 5
    assert len(wf_models._seed_list({"id": "n", "adapter": "neural"}, n_seeds=10)) == 10
    assert len(wf_models._seed_list({"id": "z", "adapter": "foundation"})) == 1
    assert len(wf_models._seed_list({"id": "f", "adapter": "foundation", "finetune": "lora"})) == 5
    assert len(wf_models._seed_list({"id": "g", "adapter": "regression", "kind": "lightgbm",
                                     "model_params": {"subsample": 0.8}})) == 5
    with pytest.raises(ValueError):
        wf_models._seed_list({"id": "n", "adapter": "neural"}, n_seeds=1)


def test_fresh_determinism_probe_uses_new_identity(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    calls = []

    def execute(view, spec):
        calls.append(spec["id"])
        return _fake_adapter(view, spec)

    monkeypatch.setattr(wf_models, "_execute_one", execute)
    frame, audit = wf_models.run(prepared, {"id": "zero", "adapter": "foundation",
                                            "verify_determinism": True})
    assert len(calls) == 2 and calls[0] != calls[1]
    assert audit["fresh_determinism_check"]["max_abs_prediction_difference"] == 0


def test_weekly_b5_h16_has_native_sigma_on_cal_and_score(tmp_path):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    context = prepared.contexts[(4, 1)].copy()
    context["target_time"] = pd.Series(
        context["target_time"].index + pd.Timedelta(hours=4),
        index=context["target_time"].index)
    prepared.contexts = {(16, 1): context}
    view = wf_models._arm_view(prepared, "EXPLORE")
    frame, audit = wf_models._phase_c_baseline(view, {"id": "B5", "baseline": "B5",
                                                 "adapter": "baseline", "arm": "EXPLORE", "seed": 42})
    assert set(frame.role) == {"cal", "score"}
    assert np.isfinite(frame[["pred", "sigma", "q10", "q50", "q90", "q95", "p_raw"]]).all().all()
    assert frame.sigma.gt(0).all() and frame.q95.gt(frame.pred).all()
    assert frame.p_raw.between(0, 1).all()
    assert audit["cells"][0]["future_perturbation_max_abs_difference"] == 0


def test_pc3_auxiliary_masks_even_week_scores_and_uses_ten_seed_namespace(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path / "pc3_finalists")
    old = prepared.contexts[(4, 0)]
    odd = pd.date_range("2021-04-12", periods=2, freq="15min")
    even = pd.date_range("2021-04-19", periods=2, freq="15min")
    scores = odd.union(even)
    contexts = {}
    for fold in (0, 1, 2):
        context = dict(old)
        all_origins = old["fit"].union(old["stop"]).union(old["cal"]).union(scores)
        context["score"] = scores
        context["x"] = pd.DataFrame({"slot7d": 20.}, index=all_origins)
        context["y"] = pd.Series(25., index=all_origins)
        context["target_time"] = pd.Series(all_origins + pd.Timedelta(hours=1), index=all_origins)
        context["d2"] = pd.Series(False, index=all_origins)
        contexts[(4, fold)] = context
    prepared.contexts = contexts
    monkeypatch.setattr(wf_models, "_execute_one", _fake_adapter)
    frame, audit = wf_models.run_pc3_confirm(prepared, {"id": "candidate", "adapter": "neural"})
    assert audit["cohort"] == "pc3_confirm" and audit["n_seeds"] == 10
    assert frame.model.eq("candidate").all()
    assert frame.loc[frame.role.eq("score"), "origin"].dt.isocalendar().week.eq(15).all()
    assert frame.loc[frame.role.eq("score"), "arm"].eq("CONFIRM").all()
    assert len(list((prepared.out / "predictions/seeds/CONFIRM/candidate__pc3").glob("*.parquet"))) == 10


def test_weekly_b1_and_b2_replay_causal_phase_c_recipes(tmp_path):
    from phase_c.data import _core_features

    root = Path(__file__).resolve().parents[2]
    prepared = WeeklyFixture(root, tmp_path)
    context = prepared.contexts[(4, 1)]
    for role in ("cal", "score"):
        origins = context[role]
        features, _ = _core_features(prepared.history, origins, 4)
        context["x"].loc[origins, "slot7d"] = features["slot7d"].to_numpy(float)
    view = wf_models._arm_view(prepared, "EXPLORE")
    for name in ("B1", "B2"):
        frame, audit = wf_models._phase_c_baseline(view, {
            "id": name, "baseline": name, "adapter": "baseline", "arm": "EXPLORE", "seed": 42})
        assert np.isfinite(frame.pred.to_numpy(float)).all()
        assert frame.model.eq(name).all()
        assert audit["cells"][0]["future_perturbation_max_abs_difference"] == 0


def test_phase_c_tcn_seed_context_restores_module_constant():
    from phase_c import tcn

    original = tcn.SEED
    with wf_models._phase_c_tcn_seed(777):
        assert tcn.SEED == 777
    assert tcn.SEED == original


def test_chronos_finetune_uses_all_sealed_week_ids():
    from phase_f.models.foundation import _fold_groups

    contexts = {(h, fold): {} for h in range(4, 17) for fold in (3, 5, 7, 9, 11)}
    assert _fold_groups(contexts, "lora") == (3, 5, 7, 9, 11)
    assert _fold_groups(contexts, None) == (-1,)
    del contexts[(16, 9)]
    with pytest.raises(ValueError, match="h16"):
        _fold_groups(contexts, "full")

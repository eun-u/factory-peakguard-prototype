"""Development-only, target-calendar weekly walk-forward cohorts for Phase F.

This is the revised 5.2-B primary evaluation geometry. The original Phase C
three-fold contexts and Phase F split lock remain separate comparison records.
No model fitting or score metric is performed here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from phase_c.data import (
    EXPECTED_SHA256, HORIZONS, _core_features, _d2_score_mask,
    _daily_profiles, _prepare_history, _sequences, load_history,
)
from phase_f.harness import BOUNDARY, KEY, arm_for_targets, seal_parent
from phase_f.registry import config_hash, sha256, write_json
from src.targets import point_targets


FIRST_SCORE_WEEK = pd.Timestamp("2021-04-12")
WEEK = pd.Timedelta(days=7)
QUARTER = pd.Timedelta(minutes=15)
SCORE_SLOTS_PER_WEEK = 7 * 24 * 4
# Phase C calibration is 35% of its validation block, with 65% used for score.
# For a complete 672-slot score week this retains the same cal:score ratio.
CAL_SLOTS = math.ceil(SCORE_SLOTS_PER_WEEK * .35 / .65)
FIT_SHARE_BEFORE_CAL = .80  # Phase C src.training._partition train-tail stop rule.


def _week_starts(history: pd.DataFrame, first_week: pd.Timestamp,
                 boundary: pd.Timestamp) -> list[pd.Timestamp]:
    first_week = pd.Timestamp(first_week)
    boundary = pd.Timestamp(boundary)
    if first_week != first_week.normalize() or first_week.dayofweek != 0:
        raise ValueError("First score week must begin Monday at 00:00")
    if boundary > BOUNDARY:
        raise ValueError("Weekly boundary cannot exceed the sealed development boundary")
    if history.empty or history.index.max() >= BOUNDARY:
        raise ValueError("History reaches the sealed development boundary")
    weeks = []
    start = first_week
    while start + WEEK <= boundary and start + WEEK - QUARTER <= history.index.max():
        weeks.append(start)
        start += WEEK
    return weeks


def _eligible_horizon(history: pd.DataFrame, horizon: int,
                      sequence_valid: pd.Series) -> dict:
    origins = pd.DatetimeIndex(history.index, name="origin")
    x_all, provenance_all = _core_features(history, origins, horizon)
    targets = point_targets(history, origins, horizon)
    target_time = pd.DatetimeIndex(targets.target_time)
    eligible = (x_all.notna().all(axis=1).to_numpy(dtype=bool)
                & targets.valid_target.to_numpy(dtype=bool)
                & (target_time.to_numpy() < np.datetime64(BOUNDARY))
                & sequence_valid.reindex(origins, fill_value=False).to_numpy(dtype=bool))
    selected = origins[eligible]
    if selected.empty:
        raise ValueError(f"No eligible Phase C core rows for h={horizon}")
    return {
        "x": x_all.loc[selected],
        "y": targets.loc[selected, "y"].rename("y"),
        "target_time": targets.loc[selected, "target_time"],
        "eligible": selected,
        "seq": None,  # Assigned once from the shared sequence table below.
        "provenance": {name: used.loc[selected] for name, used in provenance_all.items()},
    }


def build_weekly_contexts(
    history: pd.DataFrame, *, first_week: pd.Timestamp = FIRST_SCORE_WEEK,
    boundary: pd.Timestamp = BOUNDARY, min_fit_weeks: int = 8,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[tuple[int, int], dict]:
    """Build common Phase C core cohorts; ``fold`` is a zero-based score week.

    Score membership uses the *target* calendar week. Early-week origins can
    therefore fall in the previous week. Every calibration target precedes
    even the earliest score origin. Fit/stop/cal boundaries receive the same
    strict horizon target-time embargo as Phase C. The first candidate week
    is skipped uniformly for all horizons if any fit span is under eight weeks.
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("History needs interval-end DatetimeIndex")
    if "quality_bad" not in history:
        history, _ = _prepare_history(history)
    if (history.index.has_duplicates or not history.index.is_monotonic_increasing
            or history.index.max() >= BOUNDARY):
        raise ValueError("History must be sorted, unique, and development-only")
    if not history.index.equals(pd.date_range(history.index.min(), history.index.max(),
                                              freq="15min", name=history.index.name)):
        raise ValueError("History must cover a complete 15-minute grid")
    if bool(history.gap_hour.any()):
        raise ValueError("Retrospective gap masks cannot enter weekly forecast features")
    if min_fit_weeks < 1 or not horizons or any(h not in HORIZONS for h in horizons):
        raise ValueError("Invalid fit history or horizon selection")
    weeks = _week_starts(history, first_week, boundary)
    if not weeks:
        raise ValueError("No complete target-calendar score week before boundary")

    sequences = _sequences(history)
    sequence_valid = pd.Series(np.isfinite(sequences.to_numpy(dtype=float)).all(axis=1),
                               index=sequences.index)
    profiles, profile_audit = _daily_profiles(history)
    base = {h: _eligible_horizon(history, h, sequence_valid) for h in horizons}
    for data in base.values():
        data["seq"] = sequences.loc[data["eligible"]]
    contexts: dict[tuple[int, int], dict] = {}
    skipped: list[dict] = []

    for week_start in weeks:
        week_end = week_start + WEEK
        week_contexts = {}
        reason = None
        for h in horizons:
            data = base[h]
            eligible = data["eligible"]
            target_time = data["target_time"]
            target_index = pd.DatetimeIndex(target_time)
            score = eligible[(target_index >= week_start) & (target_index < week_end)]
            if score.empty:
                reason = f"h={h}: no eligible score targets"
                break
            # The first h origins can precede Monday 00:00. Calibration must
            # finish before that first *origin*, never merely before Monday.
            preweek = eligible[target_index < score.min()]
            if len(preweek) <= CAL_SLOTS + 200:
                reason = f"h={h}: insufficient preweek fit/stop/cal origins"
                break
            cal = preweek[-CAL_SLOTS:]
            train = preweek[:-CAL_SLOTS]
            # Phase C's 80/20 train-tail stop allocation, then purge labels.
            stop_start = int(len(train) * FIT_SHARE_BEFORE_CAL)
            stop = train[stop_start:]
            fit = train[:stop_start]
            fit = fit[pd.DatetimeIndex(target_time.loc[fit]) < stop.min()]
            stop = stop[pd.DatetimeIndex(target_time.loc[stop]) < cal.min()]
            cal = cal[pd.DatetimeIndex(target_time.loc[cal]) < score.min()]
            if min(map(len, (fit, stop, cal, score))) == 0:
                reason = f"h={h}: empty partition after target-time purge"
                break
            fit_span = pd.Timestamp(target_time.loc[fit].max()) - pd.Timestamp(target_time.loc[fit].min())
            if fit_span < min_fit_weeks * WEEK:
                reason = f"h={h}: fit span {fit_span} below {min_fit_weeks} weeks"
                break
            if not (target_time.loc[fit].max() < stop.min()
                    and target_time.loc[stop].max() < cal.min()
                    and target_time.loc[cal].max() < score.min()):
                raise AssertionError("Weekly target-time embargo failed")
            tau = float(np.quantile(data["y"].loc[fit].to_numpy(dtype=float), .95))
            d2, d2_audit = _d2_score_mask(profiles, target_time, fit, score)
            if not np.isfinite(tau):
                raise AssertionError("Nonfinite fit-only weekly peak threshold")
            summary = {
                "horizon_quarters": h, "week_start": str(week_start),
                "week_end_exclusive": str(week_end),
                "iso_year": int(week_start.isocalendar().year),
                "iso_week": int(week_start.isocalendar().week),
                "arm": str(arm_for_targets([week_start])[0]),
                "fit_count": len(fit), "stop_count": len(stop),
                "cal_count": len(cal), "score_count": len(score),
                "target_slots_in_week": SCORE_SLOTS_PER_WEEK,
                "cal_slots_requested": CAL_SLOTS,
                "cal_to_score_ratio_rule": "ceil(672 * .35 / .65)",
                "fit_share_before_cal": FIT_SHARE_BEFORE_CAL,
                "min_fit_weeks": min_fit_weeks,
                "fit_first": str(fit.min()),
                "fit_last_target": str(target_time.loc[fit].max()),
                "stop_first": str(stop.min()),
                "stop_last_target": str(target_time.loc[stop].max()),
                "cal_first": str(cal.min()),
                "cal_last_target": str(target_time.loc[cal].max()),
                "score_first_origin": str(score.min()),
                "score_last_target": str(target_time.loc[score].max()),
                "embargo_minutes": int(h * 15),
                "eligible_origins": len(eligible),
                **profile_audit, **d2_audit,
            }
            week_contexts[h] = {
                "x": data["x"], "y": data["y"],
                "target_time": target_time,
                "seq": data["seq"],
                "fit": fit, "stop": stop, "cal": cal, "score": score,
                "tau": tau, "d2": d2, "summary": summary,
                "provenance": data["provenance"],
            }
        if reason is not None:
            skipped.append({"week_start": str(week_start), "reason": reason})
            continue
        week_index = len({index for _, index in contexts})
        for h, context in week_contexts.items():
            context["summary"]["fold"] = week_index
            contexts[(h, week_index)] = context
    if not contexts:
        raise ValueError(f"No weekly folds meet prehistory and cohort requirements: {skipped}")
    # Expose skipped first-week reasons without changing the fit/score contract.
    for context in contexts.values():
        context["summary"]["skipped_candidate_weeks"] = skipped
    return contexts


def _score_keys(contexts: dict[tuple[int, int], dict]) -> pd.DataFrame:
    frames = []
    for (h, fold), context in sorted(contexts.items()):
        origins = context["score"]
        target_time = pd.DatetimeIndex(context["target_time"].loc[origins])
        frames.append(pd.DataFrame({
            "horizon": h, "fold": fold, "origin": origins,
            "target_time": target_time,
            "arm": arm_for_targets(target_time),
        }))
    keys = pd.concat(frames, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if keys.duplicated(KEY).any() or not keys.groupby("fold").arm.nunique().eq(1).all():
        raise AssertionError("Weekly score key or arm collision")
    return keys


class WeeklyPrepared:
    """Load sealed history once and persist a fail-closed v2 weekly split lock."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.out = self.root / "outputs/phase_f/walkforward_v2"
        self.out.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.root / "outputs/phase_f/models/walkforward_v2"
        self.seal = seal_parent(self.root)
        self.history, audit = load_history(self.root)
        self.contexts = build_weekly_contexts(self.history)
        self.keys = _score_keys(self.contexts)
        self._key_index = pd.MultiIndex.from_frame(self.keys[KEY])
        self.baselines = pd.DataFrame()  # Populated by paired weekly model runs.
        weeks = []
        for fold in sorted(set(index for _, index in self.contexts)):
            c = self.contexts[(HORIZONS[0], fold)]
            summary = c["summary"]
            weeks.append({name: summary[name] for name in (
                "fold", "week_start", "week_end_exclusive", "iso_year", "iso_week", "arm")})
        counts = [{"horizon": h, "fold": f, "fit": len(c["fit"]),
                   "stop": len(c["stop"]), "cal": len(c["cal"]),
                   "score": len(c["score"]), "d2_novel": int(c["d2"].loc[c["score"]].sum())}
                  for (h, f), c in sorted(self.contexts.items())]
        source = {name: sha256(self.root / name) for name in (
            "phase_c/data.py", "src/training.py", "phase_f/wf_harness.py")}
        record = {
            "version": 2, "scope": "Phase F walk-forward development evaluation",
            "boundary": str(BOUNDARY), "raw_source_sha256": EXPECTED_SHA256,
            "parent_seal_sha256": config_hash(self.seal["files"]),
            "source_sha256": source,
            "cohort": "Phase C finite 15 core features, valid target, complete causal 96-slot context",
            "split_by": "target_calendar_ISO_week", "even": "EXPLORE", "odd": "CONFIRM",
            "first_candidate_week": str(FIRST_SCORE_WEEK),
            "score_slots_per_calendar_week": SCORE_SLOTS_PER_WEEK,
            "cal_slots_requested": CAL_SLOTS,
            "cal_to_score_ratio_rule": "ceil(672 * .35 / .65)",
            "fit_share_before_cal": FIT_SHARE_BEFORE_CAL,
            "min_fit_weeks": 8,
            "weeks": weeks, "counts": counts,
            "skipped_candidate_weeks": self.contexts[(HORIZONS[0], 0)]["summary"]["skipped_candidate_weeks"],
            "score_key_sha256": config_hash(self.keys.astype(str).to_dict("records")),
            "historical_final_artifact_read": False, "holdout_read": False,
            "independent_unseen_validation": False,
            "prior_score_use": "Phase C and Phase E development evaluation",
            "development_history_audit": audit,
        }
        record["lock_sha256"] = config_hash(record)
        self.split_lock = record
        write_json(self.root / "outputs/phase_f/logs/walkforward_lock.json", record, exclusive=True)
        write_json(self.out / "walkforward_lock.json", record, exclusive=True)
        keys_path = self.out / "score_keys.parquet"
        if keys_path.exists():
            existing = pd.read_parquet(keys_path)
            if not existing.equals(self.keys):
                raise RuntimeError("Weekly score keys differ from locked cohort")
        else:
            self.keys.to_parquet(keys_path, index=False)

    def origins(self, horizon: int, fold: int, role: str) -> pd.DatetimeIndex:
        if role not in {"fit", "stop", "cal", "score"}:
            raise ValueError(f"Unknown weekly role: {role}")
        return self.contexts[(horizon, fold)][role]

    def validate_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Require one model's exact weekly cal and score origin coverage."""
        required = {"model", "role", "horizon", "fold", "origin", "target_time",
                    "y", "pred", "tau", "d2", "arm"}
        if not required <= set(frame.columns):
            raise ValueError(f"Weekly predictions lack columns: {sorted(required - set(frame.columns))}")
        if frame.model.nunique() != 1 or frame.duplicated(["model", "role", *KEY]).any():
            raise ValueError("Weekly predictions require one model and unique keys")
        if not set(frame.role).issubset({"cal", "score"}):
            raise ValueError("Weekly predictions include an unsupported role")
        expected_rows = sum(len(c[role]) for c in self.contexts.values()
                            for role in ("cal", "score"))
        if len(frame) != expected_rows:
            raise ValueError("Weekly calibration or score coverage changed")
        if not np.isfinite(frame.pred.to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite weekly prediction")
        score = frame.loc[frame.role.eq("score")].sort_values(KEY)
        if not pd.MultiIndex.from_frame(score[KEY]).equals(self._key_index):
            raise ValueError("Weekly score cohort changed")
        for (h, fold), context in self.contexts.items():
            for role in ("cal", "score"):
                rows = frame.loc[frame.horizon.eq(h) & frame.fold.eq(fold) & frame.role.eq(role)]
                origins = pd.DatetimeIndex(rows.sort_values("origin").origin, name="origin")
                expected = context[role]
                if not origins.equals(expected):
                    raise ValueError(f"Weekly {role} origins differ for h={h}, week={fold}")
                rows = rows.set_index("origin").loc[expected]
                target = pd.DatetimeIndex(context["target_time"].loc[expected])
                if not pd.DatetimeIndex(rows.target_time).equals(target):
                    raise ValueError("Weekly prediction target times changed")
                if not np.array_equal(rows.y.to_numpy(float), context["y"].loc[expected].to_numpy(float)):
                    raise ValueError("Weekly prediction truth changed")
                if not np.array_equal(rows.tau.to_numpy(float), np.repeat(context["tau"], len(rows))):
                    raise ValueError("Weekly fit-only tau changed")
                if rows.d2.isna().any() or not rows.d2.isin((True, False)).all():
                    raise ValueError("Weekly D2 status is incomplete")
                if not np.array_equal(rows.d2.to_numpy(bool), context["d2"].loc[expected].to_numpy(bool)):
                    raise ValueError("Weekly D2 cohort changed")
                expected_arm = arm_for_targets(target) if role == "score" else np.repeat("CAL", len(rows))
                if not np.array_equal(rows.arm.to_numpy(), expected_arm):
                    raise ValueError("Weekly arm changed")
        if pd.DatetimeIndex(frame.target_time).max() >= BOUNDARY:
            raise ValueError("Weekly target crosses sealed boundary")
        return frame

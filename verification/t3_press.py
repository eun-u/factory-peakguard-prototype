"""Task ③ press-pump feasibility checks. Run from the repository root or directly.

All fitted objects and alarm thresholds use healthy training segments only.  Segment
bootstrap resamples whole candidate segments, never individual sensor rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


SEED = 42
BOOTSTRAPS = 1000
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
CHANNELS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
TIME_ALIASES = ("TimeStamp", "timestamp", "datetime", "DateTime", "time", "Time")
LABEL_ALIASES = ("Equipment_state", "equipment_state", "label", "Label", "state")
RNG = np.random.default_rng(SEED)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _save_json(path, value):
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def _publish(result, section):
    """Stable handoff for the combined report."""
    criteria = result["warnings"]
    if result["status"] == "ok":
        raw = next(r for r in result["p2"] if r["representation"] == "raw" and r["model"] == "Mahalanobis")
        relation = next(r for r in result["p3"] if r["score"] == "combined")
        headline = (f"raw Mahalanobis AUROC {raw['auroc']:.3f} "
                    f"(95% CI {raw['auroc_ci_low']:.3f}–{raw['auroc_ci_high']:.3f}); "
                    f"combined relation AUROC {relation['auroc']:.3f} "
                    f"(95% CI {relation['auroc_ci_low']:.3f}–{relation['auroc_ci_high']:.3f})")
    else:
        headline = "자료 부재 또는 평가 불가"
    _save_json(OUT / "03_summary.json", {"status": result["status"], "criteria": criteria,
                                        "headline": headline, "reason": result.get("reason"),
                                        "sources": result.get("sources", []),
                                        "warning_operands": result.get("warning_operands", {})})
    (OUT / "03_section.md").write_text(section, encoding="utf-8")


def _read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Cannot read {path}: {last_error}")


def _discover():
    """Use schema plus a binary label or class-specific filename; never infer a class from level."""
    candidates = []
    rejected = []
    source_dir = ROOT / "data" / "raw" / "task03_press"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Task ③ raw data directory is missing: {source_dir}")
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".csv", ".parquet", ".xlsx", ".xls"):
            continue
        if any(p in (".venv", ".git", "verification", "__pycache__") for p in path.parts):
            continue
        try:
            data = _read_table(path)
        except Exception as exc:
            rejected.append(f"{path.relative_to(ROOT)}: read failure ({exc})")
            continue
        if not set(CHANNELS).issubset(data.columns):
            continue
        time_col = next((c for c in TIME_ALIASES if c in data.columns), None)
        if time_col is None:
            rejected.append(f"{path.relative_to(ROOT)}: press columns but no recognized time column")
            continue
        label_col = next((c for c in LABEL_ALIASES if c in data.columns), None)
        filename = str(path.relative_to(ROOT)).lower()
        file_class = None
        if any(s in filename for s in ("abnormal", "anomaly", "fault", "failure", "이상")):
            file_class = 1
        elif any(s in filename for s in ("normal", "healthy", "정상")):
            file_class = 0
        if label_col is None and file_class is None:
            rejected.append(f"{path.relative_to(ROOT)}: press columns but no class label or class filename")
            continue
        keep = [time_col, *CHANNELS] + ([label_col] if label_col else [])
        data = data[keep].copy()
        data = data.rename(columns={time_col: "timestamp"})
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        for col in CHANNELS:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if label_col:
            data["label"] = pd.to_numeric(data[label_col], errors="coerce")
            bad = data["label"].dropna().loc[lambda s: ~s.isin([0, 1])]
            if len(bad):
                rejected.append(f"{path.relative_to(ROOT)}: nonbinary label values present")
                continue
        else:
            data["label"] = file_class
        if file_class is not None and label_col and data["label"].dropna().ne(file_class).any():
            rejected.append(f"{path.relative_to(ROOT)}: filename class conflicts with row labels")
            continue
        data["source"] = str(path.relative_to(ROOT))
        candidates.append(data[["timestamp", *CHANNELS, "label", "source"]])
    if not candidates:
        return None, rejected
    frame = pd.concat(candidates, ignore_index=True)
    return frame, rejected


def _ci(values):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))] if len(arr) else [None, None]


def _binary_metric(y, scores, kind="auroc", ids=None):
    """Stratified segment bootstrap, keeping each labeled candidate segment intact."""
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    mask = np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(np.unique(y)) != 2:
        return {"value": None, "ci95": [None, None], "n": len(y)}
    metric = roc_auc_score if kind == "auroc" else average_precision_score
    value = float(metric(y, scores))
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    boot = []
    for _ in range(BOOTSTRAPS):
        pick = np.r_[RNG.choice(neg, len(neg), replace=True), RNG.choice(pos, len(pos), replace=True)]
        boot.append(metric(y[pick], scores[pick]))
    return {"value": value, "ci95": _ci(boot), "n": len(y), "bootstrap_unit": "candidate segment"}


def _estimate(values, method="mean"):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"value": None, "ci95": [None, None], "n": 0}
    fn = np.mean if method == "mean" else np.median
    boot = [fn(RNG.choice(vals, len(vals), replace=True)) for _ in range(BOOTSTRAPS)]
    return {"value": float(fn(vals)), "ci95": _ci(boot), "n": int(len(vals)), "bootstrap_unit": "candidate segment"}


def _paired_auc(healthy, injected):
    """Resample an original candidate segment together with its injection."""
    healthy, injected = np.asarray(healthy, float), np.asarray(injected, float)
    if len(healthy) != len(injected) or not len(healthy):
        return {"value": None, "ci95": [None, None]}
    labels = np.r_[np.zeros(len(healthy)), np.ones(len(healthy))]
    value = float(roc_auc_score(labels, np.r_[healthy, injected]))
    boot = []
    for _ in range(BOOTSTRAPS):
        pick = RNG.choice(len(healthy), len(healthy), replace=True)
        boot.append(roc_auc_score(labels, np.r_[healthy[pick], injected[pick]]))
    return {"value": value, "ci95": _ci(boot), "n_pairs": len(healthy),
            "bootstrap_unit": "original candidate segment and its injection"}


def _segment(frame):
    clean = frame.dropna(subset=["timestamp", "label", *CHANNELS]).copy()
    clean = clean[clean.label.isin([0, 1])]
    clean.label = clean.label.astype(int)
    # Deliberately avoid bridging file boundaries. Gap cut is five times pooled
    # positive median sampling interval, an empirical heuristic reported in output.
    deltas = []
    for _, part in clean.groupby(["source", "label"], sort=False):
        d = part.sort_values("timestamp").timestamp.diff().dt.total_seconds().dropna()
        deltas.extend(d[d > 0].tolist())
    if not deltas:
        raise ValueError("No positive timestamp interval: candidate segments cannot be identified")
    nominal = float(np.median(deltas))
    gap_limit = 5 * nominal
    segment_rows = []
    gap_rows = []
    segment_id = 0
    for (source, label), part in clean.groupby(["source", "label"], sort=False):
        part = part.sort_values("timestamp").reset_index(drop=True)
        gap = part.timestamp.diff().dt.total_seconds()
        for seconds in gap.dropna():
            gap_rows.append({"source": source, "label": label, "gap_seconds": float(seconds)})
        starts = (gap.isna() | (gap > gap_limit)).cumsum()
        for _, piece in part.groupby(starts, sort=False):
            segment_rows.append({"segment_id": segment_id, "source": source, "label": int(label),
                                 "start": piece.timestamp.iloc[0], "end": piece.timestamp.iloc[-1],
                                 "n_samples": len(piece), "duration_seconds": float((piece.timestamp.iloc[-1] - piece.timestamp.iloc[0]).total_seconds() + nominal),
                                 "values": piece[CHANNELS].to_numpy(float), "timestamps": piece.timestamp.to_numpy()})
            segment_id += 1
    segments = pd.DataFrame(segment_rows).sort_values(["label", "start", "source"]).reset_index(drop=True)
    gaps = pd.DataFrame(gap_rows)
    per_class_intervals = {}
    for label in (0, 1):
        vals = [v["gap_seconds"] for v in gap_rows if v["label"] == label and v["gap_seconds"] > 0]
        per_class_intervals[str(label)] = ({str(q): float(np.quantile(vals, q)) for q in (0, .5, .9, .99, 1)}
                                           if vals else {})
    diagnostics = {"initial_rows": len(frame), "usable_rows": len(clean), "dropped_rows": len(frame) - len(clean),
                   "nominal_sample_seconds": nominal, "gap_threshold_seconds": gap_limit,
                   "gap_rule": "new candidate segment when within-file/class timestamp gap > 5 x pooled median positive interval",
                   "gap_quantiles_seconds": {str(q): float(np.quantile(deltas, q)) for q in (0, .5, .9, .95, .99, 1)},
                   "interval_quantiles_by_class_seconds": per_class_intervals,
                   "nonpositive_intervals": int(sum(v["gap_seconds"] <= 0 for v in gap_rows))}
    return segments, gaps, diagnostics


def _autocorr(x):
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _features(values, representation):
    x = np.asarray(values, float).copy()
    if representation == "center":
        x -= x.mean(axis=0)
    elif representation == "z":
        x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-9)
    elif representation == "amplitude":
        x /= np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-9)
    features = []
    for j in range(3):
        v = x[:, j]
        features.extend([np.mean(v), np.std(v), np.min(v), np.quantile(v, .25),
                         np.median(v), np.quantile(v, .75), np.max(v), _autocorr(v)])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_matrix(segments, representation):
    return np.array([_features(v, representation) for v in segments["values"]], dtype=float)


class _Model:
    def __init__(self, name):
        self.name = name

    def fit(self, x):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if self.name == "IsolationForest":
            self.model = IsolationForest(n_estimators=200, random_state=SEED, contamination="auto").fit(z)
        elif self.name == "OneClassSVM":
            self.model = OneClassSVM(nu=.05, gamma="scale").fit(z)
        else:
            self.center = np.mean(z, axis=0)
            cov = np.cov(z, rowvar=False) if len(z) > 1 else np.eye(z.shape[1])
            cov = np.atleast_2d(cov)
            # Fixed ridge handles rank deficiency; it is not tuned on test.
            self.inv = np.linalg.pinv(cov + .1 * np.eye(cov.shape[0]))
        return self

    def score(self, x):
        z = self.scaler.transform(x)
        if self.name in ("IsolationForest", "OneClassSVM"):
            return -self.model.score_samples(z)
        residual = z - self.center
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", residual, self.inv, residual), 0))


class _Relation:
    def fit(self, segments):
        # No lag: pointwise coupling is identifiable without assuming how gaps
        # relate to physical cycles. Train residual scale on healthy rows only.
        x = np.concatenate(segments["values"].tolist(), axis=0)
        self.pairs = ((2, 0), (2, 1), (0, 1))
        self.models = []
        self.scale = []
        for inp, out in self.pairs:
            model = Ridge(alpha=1.0).fit(x[:, [inp]], x[:, out])
            residual = x[:, out] - model.predict(x[:, [inp]])
            self.models.append(model)
            self.scale.append(max(float(np.std(residual)), 1e-9))
        return self

    def score(self, segments):
        rows = []
        for x in segments["values"]:
            parts = []
            for (inp, out), model, scale in zip(self.pairs, self.models, self.scale):
                residual = (x[:, out] - model.predict(x[:, [inp]])) / scale
                parts.append(float(np.sqrt(np.mean(residual ** 2))))
            rows.append(parts + [float(np.sqrt(np.mean(np.square(parts))))])
        return np.asarray(rows)


def _p0(segments, gaps, diagnostics):
    by_file = []
    for (source, label), part in segments.groupby(["source", "label"]):
        by_file.append({"source": source, "label": int(label), "first": str(part.start.min()), "last": str(part.end.max()),
                        "segment_count": len(part), "sample_count": int(part.n_samples.sum()),
                        "segment_length_samples": {str(q): float(part.n_samples.quantile(q)) for q in (0, .25, .5, .75, 1)}})
    sample_rows = np.concatenate(segments["values"].tolist(), axis=0)
    waveform = {}
    for j, name in enumerate(CHANNELS):
        col = sample_rows[:, j]
        ac = [_autocorr(x[:, j]) for x in segments["values"]]
        waveform[name] = {"negative_fraction": float(np.mean(col < 0)), "minimum": float(np.min(col)),
                          "maximum": float(np.max(col)), "quantiles": {str(q): float(np.quantile(col, q)) for q in (0, .01, .25, .5, .75, .99, 1)},
                          "median_segment_lag1_autocorrelation": float(np.nanmedian(ac)) if np.isfinite(ac).any() else None}
    vibration_inference = {}
    for name in CHANNELS[:2]:
        vibration_inference[name] = (
            "추정: 음수 존재는 부호 있는 파형/중심화 측정과 일치하나 원파형 확증은 아님"
            if waveform[name]["negative_fraction"] > 0 else
            "추정: 음수가 없어 RMS/절대값 집계와 일치할 수 있으나 단극성 원신호도 가능")
    output = {**diagnostics, "files": by_file, "channels": waveform,
              "waveform_inference": vibration_inference,
              "waveform_limit": "부호·분포·자기상관·표본 간격만으로 원파형/집계값을 확정할 수 없음"}
    pd.DataFrame([{k: v for k, v in row.items() if k != "values" and k != "timestamps"} for row in segments.to_dict("records")]).to_csv(OUT / "03_segments.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUT / "03_gaps.csv", index=False, encoding="utf-8-sig")
    return output


def _p1(segments):
    rows = []
    labels = segments.label.to_numpy(int)
    for j, name in enumerate(CHANNELS):
        for title, calc in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            feat = np.array([calc(x[:, j]) for x in segments["values"]])
            forward = roc_auc_score(labels, feat)
            direction = 1 if forward >= .5 else -1
            result = _binary_metric(labels, direction * feat)
            rows.append({"feature": f"{name}_{title}", "direction": direction, "auroc": result["value"],
                         "ci_low": result["ci95"][0], "ci_high": result["ci95"][1],
                         "warning": "orientation chosen using both labels; diagnostic only"})
    feat = segments.n_samples.to_numpy(float)
    direction = 1 if roc_auc_score(labels, feat) >= .5 else -1
    result = _binary_metric(labels, direction * feat)
    rows.append({"feature": "segment_length_samples", "direction": direction, "auroc": result["value"],
                 "ci_low": result["ci95"][0], "ci_high": result["ci95"][1],
                 "warning": "orientation chosen using both labels; diagnostic only"})
    table = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    table.to_csv(OUT / "03_p1_univariate.csv", index=False, encoding="utf-8-sig")
    return rows


def _p2(segments):
    normal = segments[segments.label == 0].sort_values("start").reset_index(drop=True)
    abnormal = segments[segments.label == 1].sort_values("start").reset_index(drop=True)
    n_train = int(math.floor(len(normal) * .6))
    if n_train < 3 or len(normal) - n_train < 1 or len(abnormal) < 1:
        raise ValueError(f"Insufficient candidate segments for healthy-only 60/40 evaluation: normal={len(normal)}, abnormal={len(abnormal)}")
    train, test_normal = normal.iloc[:n_train], normal.iloc[n_train:]
    test = pd.concat([test_normal, abnormal], ignore_index=True)
    y = test.label.to_numpy(int)
    rows = []
    model_map = {}
    for representation in ("raw", "center", "z", "amplitude"):
        xtrain = _feature_matrix(train, representation)
        xtest = _feature_matrix(test, representation)
        for name in ("IsolationForest", "OneClassSVM", "Mahalanobis"):
            model = _Model(name).fit(xtrain)
            score = model.score(xtest)
            auc = _binary_metric(y, score)
            ap = _binary_metric(y, score, "pr_auc")
            rows.append({"representation": representation, "model": name, "auroc": auc["value"],
                         "auroc_ci_low": auc["ci95"][0], "auroc_ci_high": auc["ci95"][1],
                         "pr_auc": ap["value"], "pr_auc_ci_low": ap["ci95"][0], "pr_auc_ci_high": ap["ci95"][1]})
            model_map[(representation, name)] = model
    pd.DataFrame(rows).to_csv(OUT / "03_p2_models.csv", index=False, encoding="utf-8-sig")
    # Precommitted reference avoids selecting a model using the abnormal test.
    level_model = model_map[("raw", "Mahalanobis")]
    return rows, train, test_normal, abnormal, test, model_map, level_model


def _p3(train, test, level_model):
    relation = _Relation().fit(train)
    rscore = relation.score(test)
    y = test.label.to_numpy(int)
    names = ("AI2_to_AI0", "AI2_to_AI1", "AI0_to_AI1", "combined")
    rows = []
    for j, name in enumerate(names):
        auc = _binary_metric(y, rscore[:, j])
        ap = _binary_metric(y, rscore[:, j], "pr_auc")
        rows.append({"score": name, "auroc": auc["value"], "auroc_ci_low": auc["ci95"][0],
                     "auroc_ci_high": auc["ci95"][1], "pr_auc": ap["value"],
                     "pr_auc_ci_low": ap["ci95"][0], "pr_auc_ci_high": ap["ci95"][1]})
    pd.DataFrame(rows).to_csv(OUT / "03_p3_relation.csv", index=False, encoding="utf-8-sig")
    level = level_model.score(_feature_matrix(test, "raw"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in ((0, "#3b78a5"), (1, "#c43d3d")):
        mask = y == label
        ax.scatter(level[mask], rscore[mask, 3], label="normal" if label == 0 else "abnormal", alpha=.75, c=color)
    ax.set(xlabel="Raw Mahalanobis level score", ylabel="Combined relation residual RMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_p3_level_relation.png", dpi=160)
    plt.close(fig)
    return rows, relation


def _injected(base, values):
    result = base.copy(deep=True)
    result["values"] = pd.Series(values, index=result.index)
    return result


def _p4(train, test_normal, level_model, relation):
    base = test_normal.reset_index(drop=True)
    train_rows = np.concatenate(train["values"].tolist(), axis=0)
    std = np.maximum(np.std(train_rows, axis=0), 1e-9)
    base_level = level_model.score(_feature_matrix(base, "raw"))
    base_relation = relation.score(base)[:, 3]
    rows = []

    def record(kind, injected):
        scores = {"level": (base_level, level_model.score(_feature_matrix(injected, "raw"))),
                  "relation": (base_relation, relation.score(injected)[:, 3])}
        for score_name, (healthy, fake) in scores.items():
            auc = _paired_auc(healthy, fake)
            rows.append({"injection": kind, "score": score_name, "auroc": auc["value"],
                         "ci_low": auc["ci95"][0], "ci_high": auc["ci95"][1], "n_pairs": len(healthy)})

    for offset in (.5, 1.0, 2.0):
        record(f"all_channel_offset_{offset}sigma", _injected(base, [x + offset * std for x in base["values"]]))
    for scale in (1.1, 1.3):
        record(f"all_channel_scale_{scale}", _injected(base, [x * scale for x in base["values"]]))
    if len(base) >= 2:
        # Deterministic cyclic donor; no donor is the receiving segment.
        for channel in range(3):
            values = []
            for i, x in enumerate(base["values"]):
                donor = base["values"].iloc[(i + 1) % len(base)][:, channel]
                replacement = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(donor)), donor)
                copy = x.copy()
                copy[:, channel] = replacement
                values.append(copy)
            record(f"channel_shuffle_{CHANNELS[channel]}", _injected(base, values))
    pd.DataFrame(rows).to_csv(OUT / "03_p4_injections.csv", index=False, encoding="utf-8-sig")
    return rows


def _alarm_flags(scores, threshold, k, n):
    above = np.asarray(scores) > threshold
    alarm = np.zeros(len(above), dtype=bool)
    for i in range(n - 1, len(above)):
        alarm[i] = np.sum(above[i - n + 1:i + 1]) >= k
    return alarm


def _p5(train, normal, abnormal, model):
    threshold_scores = model.score(_feature_matrix(train, "raw"))
    normal_score = model.score(_feature_matrix(normal, "raw"))
    abnormal_score = model.score(_feature_matrix(abnormal, "raw"))
    normal_hours = float(normal.duration_seconds.sum() / 3600)
    rows = []
    for alpha in (.01, .05):
        threshold = float(np.quantile(threshold_scores, 1 - alpha))
        for k, n in ((1, 1), (2, 2), (2, 3), (3, 5)):
            flags_n = _alarm_flags(normal_score, threshold, k, n)
            flags_a = _alarm_flags(abnormal_score, threshold, k, n)
            events_n = flags_n & ~np.r_[False, flags_n[:-1]]
            transitions = int(np.sum(np.diff(np.r_[False, flags_n, False].astype(int)) != 0))
            first = np.flatnonzero(flags_a)
            delay_segments = int(first[0]) if len(first) else None
            delay_seconds = float((abnormal.start.iloc[first[0]] - abnormal.start.iloc[0]).total_seconds()) if len(first) else None
            # Segment bootstrap for rates. Chattering is sequence-dependent;
            # report the observed count without a misleading reordering CI.
            far = _estimate(flags_n.astype(float))
            dr = _estimate(flags_a.astype(float))
            counts_per_hour = float(events_n.sum() / normal_hours) if normal_hours > 0 else None
            rate_boot = []
            if normal_hours > 0:
                dur = normal.duration_seconds.to_numpy(float) / 3600
                for _ in range(BOOTSTRAPS):
                    pick = RNG.choice(len(flags_n), len(flags_n), replace=True)
                    denom = dur[pick].sum()
                    rate_boot.append(events_n[pick].sum() / denom if denom > 0 else np.nan)
            rows.append({"alpha": alpha, "threshold": threshold, "rule": f"{k}/{n}",
                         "normal_false_alarm_fraction": far["value"], "false_alarm_ci95": far["ci95"],
                         "false_alarm_events": int(events_n.sum()), "alarms_per_operating_hour": counts_per_hour,
                         "alarms_per_hour_ci95": _ci(rate_boot),
                         "chattering_transitions": transitions, "abnormal_segment_detection_fraction": dr["value"],
                         "detection_ci95": dr["ci95"], "delay_segments_from_abnormal_start": delay_segments,
                         "delay_seconds_from_abnormal_start": delay_seconds, "delay_ci95": [None, None]})
    pd.DataFrame([{**r, "false_alarm_ci95": str(r["false_alarm_ci95"]), "alarms_per_hour_ci95": str(r["alarms_per_hour_ci95"]), "detection_ci95": str(r["detection_ci95"])} for r in rows]).to_csv(
        OUT / "03_p5_alarms.csv", index=False, encoding="utf-8-sig")
    return rows


def _warning(value):
    return "해당" if value else "비해당"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, rejected = _discover()
    if frame is None or frame.label.dropna().nunique() != 2:
        reason = "No readable, timestamped, three-channel press data with both labeled normal and abnormal rows were found."
        result = {"status": "unavailable", "reason": reason, "rejected": rejected,
                  "warnings": {"K3-a": "판정불가", "K3-b": "판정불가", "K3-c": "판정불가"}}
        _save_json(OUT / "03_press.json", result)
        section = "# ③ 사전 타당성 검증\n\n현재 상태: **판정불가**. " + reason + "\n\nP0~P5는 데이터 부재로 실행하지 못했습니다. 수치를 추정하거나 대체 데이터로 채우지 않았습니다.\n"
        (OUT / "03_press.md").write_text(section, encoding="utf-8")
        _publish(result, section)
        return result
    try:
        segments, gaps, diagnostics = _segment(frame)
        if segments.label.nunique() != 2:
            raise ValueError("After dropping missing rows, both classes no longer have candidate segments")
        p0 = _p0(segments, gaps, diagnostics)
        p1 = _p1(segments)
        p2, train, test_normal, abnormal, test, model_map, level_model = _p2(segments)
        p3, relation = _p3(train, test, level_model)
        p4 = _p4(train, test_normal, level_model, relation)
        p5 = _p5(train, test_normal, abnormal, level_model)
        level_univariate = max((r["auroc"] for r in p1 if r["feature"].endswith(("_mean", "_std"))), default=None)
        normalized = max((r["auroc"] for r in p2 if r["representation"] in ("center", "z", "amplitude")), default=None)
        residual = next(r["auroc"] for r in p3 if r["score"] == "combined")
        shuffled = [r["auroc"] for r in p4 if r["injection"].startswith("channel_shuffle") and r["score"] == "relation"]
        length = next(r["auroc"] for r in p1 if r["feature"] == "segment_length_samples")
        warnings = {
            "K3-a": _warning(level_univariate >= .95 and normalized <= .70 and residual <= .70) if normalized is not None else "판정불가",
            "K3-b": _warning(max(shuffled) < .80) if shuffled else "판정불가",
            "K3-c": _warning(length >= .90),
        }
        result = {"status": "ok", "sources": sorted(frame.source.unique().tolist()), "rejected": rejected,
                  "p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
                  "warnings": warnings, "warning_operands": {"best_mean_std_auroc": level_univariate,
                      "best_normalized_auroc": normalized, "combined_residual_auroc": residual,
                      "best_shuffle_relation_auroc": max(shuffled) if shuffled else None,
                      "segment_length_auroc": length},
                  "limitations": ["Candidate segments are defined by timestamp gaps, not verified press cycles.",
                                  "Abnormal segments may belong to one recording session; segment CIs do not capture independent-failure uncertainty.",
                                  "Normalized performance loss can reflect session shift or genuine fault-related level change.",
                                  "No fault onset or return-to-normal timestamp is verified; release time is unmeasurable.",
                                  "P1 score orientation uses both labels for diagnosis and is not a deployable classifier.",
                                  "P5 detection delay is from abnormal file start; it is not validated early-warning lead time.",
                                  "Channel shuffle interpolates donor segments when lengths differ; this also alters shape and autocorrelation.",
                                  "Sequential chatter and one-session detection delay lack a defensible independent-event bootstrap CI."]}
        _save_json(OUT / "03_press.json", result)
        def metric_str(value, low, high):
            return f"{value:.3f} (95% CI {low:.3f}–{high:.3f})"

        best_single = max(p1, key=lambda row: row["auroc"])
        raw_ref = next(r for r in p2 if r["representation"] == "raw" and r["model"] == "Mahalanobis")
        combined = next(r for r in p3 if r["score"] == "combined")
        lines = ["# ③ 사전 타당성 검증", "", "상태: 실행 완료. 세부 수치는 `03_press.json` 및 아래 CSV를 참조.", "",
                 "## P0 구조 진단", f"- 사용 행: {p0['usable_rows']} / 전체 {p0['initial_rows']}; candidate segment 수: {len(segments)}",
                 f"- 표본 간격 중앙값: {p0['nominal_sample_seconds']:.6g}초; gap 기준: {p0['gap_threshold_seconds']:.6g}초 (중앙값의 5배).",
                 f"- 정상/이상 segment 수: {len(train)+len(test_normal)}/{len(abnormal)}. 측정 범위와 간격 분포는 `03_press.json`, `03_gaps.csv` 참조.",
                 "- 원파형 여부: 추정만 가능. 부호·분포·자기상관은 JSON 참조.", "",
                 "## P1 세션 shortcut", f"- 최고 단변량: {best_single['feature']} AUROC {metric_str(best_single['auroc'], best_single['ci_low'], best_single['ci_high'])}. 라벨로 점수 방향을 정한 진단용 값이다.",
                 "- 전체 표: `03_p1_univariate.csv`.", "",
                 "## P2 정규화 불변성", f"- 사전 고정 raw Mahalanobis: AUROC {metric_str(raw_ref['auroc'], raw_ref['auroc_ci_low'], raw_ref['auroc_ci_high'])}; PR-AUC {metric_str(raw_ref['pr_auc'], raw_ref['pr_auc_ci_low'], raw_ref['pr_auc_ci_high'])}.",
                 "- 정상 segment 앞 60%로만 학습하고 뒤 40%와 모든 이상 segment로 평가. 전체 12개 조합: `03_p2_models.csv`.",
                 "- 정규화 후 하락은 세션 차이와 실제 고장 레벨 변화 양쪽으로 해석될 수 있다.", "",
                 "## P3 3채널 관계 잔차", f"- 합산 잔차 AUROC {metric_str(combined['auroc'], combined['auroc_ci_low'], combined['auroc_ci_high'])}; PR-AUC {metric_str(combined['pr_auc'], combined['pr_auc_ci_low'], combined['pr_auc_ci_high'])}.",
                 "- `03_p3_relation.csv`, 산점도 `03_p3_level_relation.png`.", "",
                 "## P4 합성 주입", "- offset·scale 및 채널 교체별 레벨/관계 AUROC: `03_p4_injections.csv`. 가짜 이상은 실제 고장 증거가 아니다.", "",
                 "## P5 경보 KPI", "- 정상 학습 점수 분위수만으로 α=1%, 5% 임계값을 설정했다. 규칙별 정상 운전시간당 경보 사건·채터링·이상 segment 탐지율·첫 경보 지연: `03_p5_alarms.csv`.",
                 "- 지연은 이상 파일 시작 기준이다. 독립 고장 시작 시각과 해제 시각이 확인되지 않아 지연 CI·해제 시간은 판정불가.", "", "## 사전 경고 기준 판정",
                 *[f"- {k}: {v}" for k, v in warnings.items()], "", "## 해석 한계",
                 *[f"- {v}" for v in result["limitations"]], ""]
        section = "\n".join(lines)
        (OUT / "03_press.md").write_text(section, encoding="utf-8")
        _publish(result, section)
        return result
    except ValueError as exc:
        result = {"status": "unavailable", "reason": str(exc), "sources": sorted(frame.source.unique().tolist()),
                  "warnings": {"K3-a": "판정불가", "K3-b": "판정불가", "K3-c": "판정불가"}}
        _save_json(OUT / "03_press.json", result)
        section = f"# ③ 사전 타당성 검증\n\n현재 상태: **판정불가**. {exc}\n"
        (OUT / "03_press.md").write_text(section, encoding="utf-8")
        _publish(result, section)
        return result


if __name__ == "__main__":
    main()

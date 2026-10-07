"""Fail-closed Stage 4 entry barrier for the absolute performance campaign.

The current integration state has no approved representative from the goal
campaign.  A signed pending policy therefore blocks *all* Stage 4 paths before
the old finalist code can read search evidence or create selection locks.  A
future implementation must explicitly add a ready-state integration; changing
the policy status alone can never open CONFIRM.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

from phase_f.goal_protocol import CONTRACT as GOAL_CONTRACT, TARGETS
from phase_f.registry import config_hash
from phase_f.wf_contract import CONTRACT as WF_CONTRACT


POLICY_PATH = Path("outputs/phase_f/logs/goal_confirmation_policy.json")
_BUDGET = {
    "minimum_gbdt_trials_each": 500,
    "maximum_wallclock_budget": None,
    "stochastic_seeds": 5,
    "relative_improvement_minimum": 0.02,
    "nonimproving_rounds": 2,
}
_POLICY = {
    "schema": 1,
    "protocol": "absolute_goal_confirmation_gate_v1",
    "status": "integration_pending",
    "absolute_targets": TARGETS,
    "absolute_goal_contract_sha256": GOAL_CONTRACT["contract_sha256"],
    "phase_f_contract_sha256": WF_CONTRACT["contract_sha256"],
    "stage0_3_budget": _BUDGET,
    "stage0_3_budget_unchanged": True,
    "finalist_cap": 7,
    "confirm_transaction_count": 1,
    "candidate_confirm_metric_read": False,
    "holdout_authorized": False,
    "holdout_read": False,
    "goal_achieved": False,
}


def pending_policy() -> dict:
    """Return the exact self-checksummed policy for the coordinator to save."""
    payload = json.loads(json.dumps(_POLICY, ensure_ascii=False))
    payload["policy_sha256"] = config_hash(payload)
    return payload


def require_stage4_entry(prepared) -> None:
    """Block a present policy before Stage 4 touches any existing artifact."""
    path = Path(prepared.root) / POLICY_PATH
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return  # Original Phase F behavior when no goal campaign is registered.
    except OSError as exc:
        raise RuntimeError("Goal confirmation policy path cannot be inspected; Stage 4 remains closed") from exc
    if not stat.S_ISREG(mode):
        raise RuntimeError("Goal confirmation policy path is not a regular file; Stage 4 remains closed")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Goal confirmation policy cannot be read; Stage 4 remains closed") from exc
    if not isinstance(record, dict):
        raise RuntimeError("Goal confirmation policy must be an object; Stage 4 remains closed")
    payload = {key: value for key, value in record.items() if key != "policy_sha256"}
    if (set(record) != set(_POLICY) | {"policy_sha256"}
            or record.get("policy_sha256") != config_hash(payload)):
        raise RuntimeError("Goal confirmation policy checksum or schema changed; Stage 4 remains closed")
    if any(WF_CONTRACT.get(key) != value for key, value in _BUDGET.items()):
        raise RuntimeError("Stage 0–3 search budget changed; Stage 4 remains closed")
    for field, expected in _POLICY.items():
        if payload.get(field) != expected or type(payload.get(field)) is not type(expected):
            raise RuntimeError(f"Goal confirmation policy {field} is unsupported; Stage 4 remains closed")
    raise RuntimeError(
        "Absolute-goal integration is pending. Keep Stage 4 closed until the completed "
        "goal experiments identify one locked representative within the seven-finalist cap; "
        "then implement and review a single shared CONFIRM transaction. Changing this "
        "policy to ready does not authorize CONFIRM. The holdout remains sealed."
    )

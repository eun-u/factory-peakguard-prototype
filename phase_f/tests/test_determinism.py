"""CPU checks for the neural adapter's process-wide determinism contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from phase_f.models import neural


@pytest.fixture
def synthetic_case():
    index = pd.date_range("2021-04-01 00:15", periods=320, freq="15min", name="ts_end")
    positions = np.arange(len(index), dtype=float)
    power = 50 + 5 * np.sin(2 * np.pi * positions / 96) + .01 * positions
    history = pd.DataFrame({"power": power, "time_repaired": False}, index=index)
    fit = pd.DatetimeIndex(index[150:176:2], name="origin")
    stop = pd.DatetimeIndex(index[190:200:2], name="origin")
    context = {
        "fit": fit, "stop": stop,
        "target_time": pd.Series(index + pd.Timedelta(hours=1), index=index),
        "y": pd.Series(power, index=index).shift(-4),
        "tau": 58., "summary": {"horizon_quarters": 4},
    }
    return history, context


@pytest.fixture(autouse=True)
def restore_caller_determinism():
    previous = _flags()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def _flags() -> tuple[bool, bool]:
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


def _config(mode: str) -> dict:
    return {"context_length": 16, "seeds": [42], "max_epochs": 1,
            "patience": 1, "batch_size": 8, "device": "cpu",
            "determinism_mode": mode}


@pytest.mark.parametrize("mode, prior", [
    ("strict", (True, True)),
    ("warn_only", (True, False)),
])
def test_fit_uses_requested_mode_and_restores_callers_flags(synthetic_case, monkeypatch, mode, prior):
    history, context = synthetic_case
    torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])
    original_loss = neural._loss
    seen = []

    def inspect_loss(*args, **kwargs):
        seen.append(_flags())
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(neural, "_loss", inspect_loss)
    bundle = neural.fit_model("dlinear", history, context, _config(mode))
    assert seen and set(seen) == {(True, mode == "warn_only")}
    assert _flags() == prior
    assert bundle["config"]["determinism_mode"] == mode
    assert bundle["determinism_mode"] == mode
    assert bundle["seeds"][0]["determinism_mode"] == mode
    assert isinstance(bundle["nondeterministic_operations"], list)


@pytest.mark.parametrize("mode, prior", [
    ("strict", (False, False)),
    ("warn_only", (True, False)),
])
def test_forced_training_error_restores_callers_flags(synthetic_case, monkeypatch, mode, prior):
    history, context = synthetic_case
    torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])
    seen = []

    def fail_after_seed(*args, **kwargs):
        seen.append(_flags())
        raise RuntimeError("forced training error after seed")

    monkeypatch.setattr(neural, "_loss", fail_after_seed)
    with pytest.raises(RuntimeError, match="forced training error after seed"):
        neural.fit_model("dlinear", history, context, _config(mode))
    assert seen == [(True, mode == "warn_only")]
    assert _flags() == prior


def test_invalid_determinism_mode_is_rejected_before_training(synthetic_case, monkeypatch):
    history, context = synthetic_case
    monkeypatch.setattr(neural, "_train_seed", lambda *_: pytest.fail("training must not start"))
    with pytest.raises(ValueError, match="determinism_mode"):
        neural.fit_model("dlinear", history, context, _config("disabled"))


def test_determinism_policy_changes_checkpoint_fingerprint(synthetic_case):
    history, context = synthetic_case
    previous = _flags()
    strict = neural.fit_model("dlinear", history, context, _config("strict"))
    warn = neural.fit_model("dlinear", history, context, _config("warn_only"))
    assert strict["fingerprint"] != warn["fingerprint"]
    assert _flags() == previous


@pytest.mark.parametrize("mode", ["strict", "warn_only"])
@pytest.mark.parametrize("fail_inside", [False, True])
def test_cuda_backend_policy_scopes_flags_without_gpu_work(mode, fail_inside):
    # torch.device("cuda") only identifies a policy branch. This test never
    # creates a CUDA tensor, queries a GPU, or launches a kernel.
    matmul = torch.backends.cuda.matmul
    modern = hasattr(matmul, "fp32_precision")
    original = (_flags(), torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
                matmul.fp32_precision if modern else matmul.allow_tf32)
    try:
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if modern:
            matmul.fp32_precision = "tf32"
        else:
            matmul.allow_tf32 = True
        prior = (_flags(), torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark,
                 matmul.fp32_precision if modern else matmul.allow_tf32)

        def inspect_policy():
            with neural._backend_policy(torch.device("cuda"), mode):
                assert _flags() == (True, mode == "warn_only")
                assert torch.backends.cudnn.deterministic is True
                assert torch.backends.cudnn.benchmark is False
                if modern:
                    assert matmul.fp32_precision == "ieee"
                else:
                    assert matmul.allow_tf32 is False
                if fail_inside:
                    raise RuntimeError("forced backend policy error")

        if fail_inside:
            with pytest.raises(RuntimeError, match="forced backend policy error"):
                inspect_policy()
        else:
            inspect_policy()
        after = (_flags(), torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark,
                 matmul.fp32_precision if modern else matmul.allow_tf32)
        assert after == prior
    finally:
        torch.use_deterministic_algorithms(original[0][0], warn_only=original[0][1])
        torch.backends.cudnn.deterministic = original[1]
        torch.backends.cudnn.benchmark = original[2]
        if modern:
            matmul.fp32_precision = original[3]
        else:
            matmul.allow_tf32 = original[3]

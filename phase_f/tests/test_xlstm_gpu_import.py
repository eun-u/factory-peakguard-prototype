"""Official xLSTM vanilla import must not require CUDA compiler installation."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("cuda_lib", [None, "preserve-existing-cuda-lib"])
def test_gpu_visible_vanilla_import_restores_compiler_configuration(cuda_lib: str | None):
    repo = Path(__file__).resolve().parents[2]
    site = repo / "outputs/phase_f/optional_envs/xlstm/site"
    if not site.is_dir():
        pytest.skip("Official xlstm==2.0.6 optional site is unavailable")
    code = textwrap.dedent("""
        import os
        import torch
        import torch.utils.cpp_extension as extension
        from phase_f.models.neural import _XLSTM

        # Emulate the CUDA-driver-visible / toolkit-absent combination on any
        # test host. No tensor or model is moved to a GPU.
        extension.CUDA_HOME = None
        torch.cuda.is_available = lambda: True
        original_lib = os.environ.get("CUDA_LIB")
        model = _XLSTM(1, 0, 16, 32, 1, 4)
        assert extension.CUDA_HOME is None
        assert os.environ.get("CUDA_LIB") == original_lib
        assert torch.cuda.is_available()
        assert model.encoder.config.slstm_block.slstm.backend == "vanilla"
        assert next(model.parameters()).device.type == "cpu"
        values = model(torch.ones(2, 16), torch.ones(2, 16), torch.empty(2, 0))
        assert values.shape == (2, 1) and torch.isfinite(values).all()
    """)
    env = os.environ.copy()
    if cuda_lib is None:
        env.pop("CUDA_LIB", None)
    else:
        env["CUDA_LIB"] = cuda_lib
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

"""Small synthetic checks; no project data or evaluation split is touched."""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase_c.tcn import _TCNDirect, fit_tcn, predict_tcn


def test_encoder_has_no_future_information():
    torch.manual_seed(7)
    model = _TCNDirect(context_dim=2, channels=32, dropout=0.1).eval()
    sequence = torch.randn(1, 1, 96)
    changed_future = sequence.clone()
    changed_future[:, :, 48:] += 1000

    with torch.inference_mode():
        original = model.encoder(sequence)
        changed = model.encoder(changed_future)

    torch.testing.assert_close(original[:, :, :48], changed[:, :, :48])
    assert not torch.allclose(original[:, :, -1], changed[:, :, -1])


def test_fit_only_statistics_and_saved_scalar_predictions():
    rng = np.random.default_rng(7)
    seq_fit = rng.normal(20, 2, size=(8, 96)).astype(np.float32)
    context_fit = rng.normal(0, 1, size=(8, 2)).astype(np.float32)
    y_fit = rng.normal(5, 0.5, size=8).astype(np.float32)
    # Deliberately shift stop distribution to expose accidental stop fitting.
    seq_stop = np.full((3, 96), 1000, dtype=np.float32)
    context_stop = np.full((3, 2), 1000, dtype=np.float32)
    y_stop = np.full(3, 1000, dtype=np.float32)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle = fit_tcn(
            seq_fit, context_fit, y_fit,
            seq_stop, context_stop, y_stop,
            {"channels": 32, "dropout": 0.1, "max_epochs": 1},
            device="cpu",
        )
        norm = bundle["normalization"]
        assert norm["seq_mean"] == pytest.approx(float(seq_fit.mean()))
        assert norm["y_mean"] == pytest.approx(float(y_fit.mean()))
        np.testing.assert_allclose(norm["context_mean"], context_fit.mean(axis=0), rtol=1e-6)
        assert bundle["metadata"]["best_epoch"] == 1
        assert len(bundle["metadata"]["history"]) == 1

        predictions = predict_tcn(bundle, seq_fit[:2], context_fit[:2], batch_size=1)
        assert predictions.shape == (2,)
        assert np.isfinite(predictions).all()

        buffer = io.BytesIO()
        torch.save(bundle, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=True)
        np.testing.assert_allclose(
            predictions,
            predict_tcn(restored, seq_fit[:2], context_fit[:2], batch_size=2),
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        torch.set_num_threads(previous_threads)

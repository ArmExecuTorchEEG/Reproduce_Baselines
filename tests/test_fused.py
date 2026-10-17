"""Fused export (src/export_fused.py): CAR + EA + input scale folded into the spatial conv."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT

from src.export_fused import car_matrix, fold_matrix, fold_model, fused_params, identity_check, parse_subjects
from src.models.tinyeegnet import TinyEEGNet
from src.quantize import float_logits

WIDE32 = {"F1": 16, "D": 2, "F2": 32, "kern_t": 32, "kern_s": 16}
FUSED = REPO_ROOT / "artifacts" / "v4_fused"
SHIPPED = REPO_ROOT / "artifacts" / "v4"
RUN = REPO_ROOT / "runs" / "v4soft" / "student_wide32_ft_seed0"


def _random_model(seed: int = 0) -> torch.nn.Module:
    """wide32 TinyEEGNet with non-trivial BatchNorm statistics and affine terms (so bn1 has a shift)."""
    torch.manual_seed(seed)
    model = TinyEEGNet(n_samples=256, **WIDE32)
    with torch.no_grad():
        for bn in (model.bn1, model.bn2, model.bn3):
            bn.running_mean.normal_(0, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.normal_(0, 0.5)
    return model.eval()


def _spd_inverse_sqrt(rng: np.random.Generator, n: int = 22) -> np.ndarray:
    """An EA-like matrix: inverse square root of a CAR'd covariance plus a ridge."""
    X = car_matrix(n) @ rng.standard_normal((n, 4000)) * rng.uniform(1, 10, (n, 1))
    R = X @ X.T / X.shape[1]
    R += 1e-3 * np.trace(R) / n * np.eye(n)
    w, V = np.linalg.eigh(R)
    return (V / np.sqrt(w)) @ V.T


def test_fold_identity_float():
    rng = np.random.default_rng(0)
    model = _random_model()
    A, scale = _spd_inverse_sqrt(rng), 0.97
    M = fold_matrix(A, scale)
    folded = fold_model(model, M)
    X = rng.standard_normal((100, 22, 256)).astype(np.float32) * 10 + rng.standard_normal((100, 1, 256)).astype(np.float32) * 20
    assert identity_check(model, folded, X, M) <= 1e-4
    # the shipped path (CAR'd window -> EA -> / scale -> model) equals the folded model on the raw window
    Xc = (car_matrix() @ X.astype(np.float64))
    shipped = float_logits(model, (np.einsum("cd,ndt->nct", A, Xc) / scale).astype(np.float32)[:, None])
    assert np.abs(shipped - float_logits(folded, X[:, None])).max() <= 1e-4
    # CAR is idempotent: referenced and un-referenced windows give the same folded output
    assert np.abs(float_logits(folded, Xc.astype(np.float32)[:, None]) - float_logits(folded, X[:, None])).max() <= 1e-4
    # only conv_spatial.weight and bn2.running_mean change
    a, b = model.state_dict(), folded.state_dict()
    changed = {k for k in a if not torch.equal(a[k], b[k])}
    assert changed == {"conv_spatial.weight", "bn2.running_mean"}


def test_fold_needs_bn2_correction():
    """Without the bn2 running-mean correction the bn1 shift breaks the identity."""
    rng = np.random.default_rng(1)
    model = _random_model(1)
    M = fold_matrix(_spd_inverse_sqrt(rng), 1.0)
    folded = fold_model(model, M)
    with torch.no_grad():
        folded.bn2.running_mean.copy_(model.bn2.running_mean)
    X = rng.standard_normal((20, 22, 256)).astype(np.float32)
    assert identity_check(model, folded, X, M) > 1e-3


def test_parse_subjects():
    assert parse_subjects("1..9") == list(range(1, 10))
    assert parse_subjects("3") == [3]
    assert parse_subjects("1,5,3") == [1, 3, 5]


def test_fused_params_schema(tmp_path):
    from src.quantize import v2_run_config

    run = tmp_path / "run"
    run.mkdir()
    (run / "student.json").write_text(json.dumps(WIDE32))
    (run / "params.json").write_text(json.dumps({
        "window_samples": 256, "n_channels": 22, "input_scale": 0.97, "seed": 0, "ea_eps": 0.001,
        "ea_matrices": {str(s): np.eye(22).tolist() for s in (1, 2)}, "student": {"name": "wide32", **WIDE32},
        "config": {"fs_model": 125, "window_start": 62, "ea_eps": 0.001, "filter": {"low": 7.0, "high": 30.0, "order": 4},
                   "test": {"calib_trials": 24}}}))
    cfg = v2_run_config(run)
    files = {s: {"file": f"tinyeegnet_fused_s{s}_int8.pte", "bytes": 1, "sha256": "0"} for s in (1, 2)}
    p = fused_params(cfg, {1: (0.1, 2), 2: (0.2, -3)}, files, "session1", "car")
    assert p["version"] == "v2-fused" and p["subjects"] == [1, 2]
    assert p["ea_matrices"] == "folded" and p["ea_matrix"] == "folded" and p["input_scale"] == 1.0
    assert p["input_quant_scale"] == {"1": 0.1, "2": 0.2} and p["input_zero_point"] == {"1": 2, "2": -3}
    assert p["reference"] == "common_average_22" and p["reference_folded_in_pte"] is True
    assert fused_params(cfg, {1: (0.1, 2)}, files, "session1", "noref")["reference"] == "folded"
    assert p["n_samples"] == 256 and p["decimation"]["factor"] == 2 and p["window"]["start_samples_after_cue_125hz"] == 62


# ----------------------------------------------------------------------------- shipped artifacts (skip if absent)
needs_fused = pytest.mark.skipif(not (FUSED / "preprocess_params_fused.json").is_file(),
                                 reason="artifacts/v4_fused not exported (python -m src.export_fused)")


@needs_fused
def test_fused_params_artifact():
    p = json.loads((FUSED / "preprocess_params_fused.json").read_text())
    ship = json.loads((SHIPPED / "preprocess_params.json").read_text()) if (SHIPPED / "preprocess_params.json").is_file() else None
    assert p["version"] == "v2-fused"
    assert p["ea_matrices"] == "folded" and p["input_scale"] == 1.0
    assert p["reference"] in ("common_average_22", "folded")
    assert set(p["input_quant_scale"]) == set(p["input_zero_point"]) == set(p["pte"]) == {str(s) for s in p["subjects"]}
    assert all(q > 0 for q in p["input_quant_scale"].values())
    assert all(-128 <= z <= 127 for z in p["input_zero_point"].values())
    if ship is not None:
        for key in ("sos", "fs", "fs_model", "n_samples", "input_shape", "channel_order", "filter", "decimation",
                    "window", "cue_offset_s", "class_map", "student"):
            assert p[key] == ship[key], key
        assert np.isclose(p["input_scale_folded"], ship["input_scale"])
    for s, rec in p["pte"].items():
        path = FUSED / rec["file"]
        assert path.stat().st_size == rec["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"]


@needs_fused
@pytest.mark.skipif(not (SHIPPED / "test_vectors.npz").is_file(), reason="artifacts/v4 not exported")
def test_fused_a03_vectors_match_shipped():
    fused, ship = np.load(FUSED / "test_vectors_s3.npz"), np.load(SHIPPED / "test_vectors.npz")
    assert fused["x"].dtype == np.int8 and fused["x"].shape == (20, 1, 22, 256)
    np.testing.assert_array_equal(fused["y"], ship["y"])
    agree = np.mean(fused["logits"].argmax(1) == ship["logits"].argmax(1))
    assert agree >= 0.95
    # the folded float model on the dequantized fused vectors agrees with the stored int8 logits
    p = json.loads((FUSED / "preprocess_params_fused.json").read_text())
    scale, zp = p["input_quant_scale"]["3"], p["input_zero_point"]["3"]
    model = TinyEEGNet(n_samples=256, **json.loads((FUSED / "student.json").read_text()))
    model.load_state_dict(torch.load(FUSED / "tinyeegnet_fused_s3_float.pt", map_location="cpu", weights_only=True))
    f = float_logits(model.eval(), (fused["x"].astype(np.float32) - zp) * np.float32(scale))
    assert np.mean(f.argmax(1) == fused["logits"].argmax(1)) >= 0.95


@needs_fused
@pytest.mark.skipif(not (RUN / "model.pt").is_file(), reason="v4soft run not present")
def test_fused_float_state_is_the_fold_of_the_run():
    from src.quantize import load_float_model, v2_run_config

    cfg = v2_run_config(RUN)
    model = load_float_model(RUN / "model.pt", cfg["window"], cfg["widths"])
    expect = fold_model(model, fold_matrix(cfg["ea_matrices"][3], cfg["input_scale"])).state_dict()
    got = torch.load(FUSED / "tinyeegnet_fused_s3_float.pt", map_location="cpu", weights_only=True)
    for k in expect:
        torch.testing.assert_close(got[k], expect[k])

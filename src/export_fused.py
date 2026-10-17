# SPDX-License-Identifier: Apache-2.0
"""Fused export: fold common average reference + Euclidean alignment + input scale into the
TinyEEGNet spatial convolution, one int8 Ethos-U85 ``.pte`` per BCI IV 2a subject.

    python -m src.export_fused --run-dir runs/v4soft/student_wide32_ft_seed0 --subjects 1..9 \\
        --out artifacts/v4_fused/

Shipped (v4) model input per window ``X`` (22 x 256 µV, band-passed and decimated):
``model(A_s C X / input_scale)`` with ``C = I - 11ᵀ/22`` (CAR; the shipped pipeline applies it
at 250 Hz before the filter, which commutes with it) and ``A_s`` the subject's EA matrix.
``conv_temporal`` applies the same kernel to every channel and ``bn1`` is per filter, so a
22 x 22 channel mix ``M`` passes through both, and the depthwise spatial weights absorb it:
``w' = Mᵀ w`` with ``M_s = A_s C / input_scale``. ``bn1``'s shift ``c_f`` reaches the spatial
conv as a constant on every channel; ``M_s 1 = 0`` (CAR), so the original model sees
``c_f wᵀ1`` there and the folded one ``c_f w'ᵀ1 = 0``: ``bn2.running_mean`` absorbs the
difference ``c_f (wᵀ1 - w'ᵀ1)`` (no other layer changes). In float the folded model takes the
band-passed, decimated µV window, referenced or not (``C`` is idempotent).

Board input (``--input``): the int8 input qparams depend on which window the board feeds.
``car`` (default) = the board keeps the CAR (step 1): PTQ and test vectors on CAR'd windows,
the int8 accuracy gate passes. ``noref`` = the board drops the CAR too: the common mode
(std ~2x the referenced signal) widens the int8 input range, and the int8 accuracy drops
(``docs/fused_export.md``).

Writes into ``--out`` per subject ``N``: ``tinyeegnet_fused_sN_float.pt`` (folded state dict),
``tinyeegnet_fused_sN_int8.pte`` (PTQ + Ethos-U lowering of :mod:`src.export`, ``ethos-u85-256``),
``test_vectors_sN.npz`` (``x`` int8 (20, 1, 22, 256) = session 2 trials 25-44 board-input
windows quantized with the subject's qparams; ``logits`` of the fake-quant model; ``y``),
``vela_sN.log``; and once ``preprocess_params_fused.json`` (``version`` ``"v2-fused"``),
``student.json``, ``fused_metrics.json`` (identity check, accuracy vs the shipped int8 model,
files). ``--no-pte`` writes only ``fused_metrics_{input}_{calib}_nopte.json``.

PTQ calibration (``--calib``): ``session1`` (default) = 200 seeded windows of the subject's
2a session 1, all four MI classes; the cue event codes are read only to locate trials and
no class label enters the calibration (the inverse of the v2 calibration
windows' map does not exist: CAR is singular, and their source session is not recorded).
``session2_calib`` = the subject's 24 EA calibration trials (session 2 trials 1-24). The test
block (session 2 trials 25-144) is never used for calibration.

Windows come from the raw 2a files (``download.load_raw``), cut exactly as
``cache125.windows_2a`` but without the CAR step, cached in ``runs/cache125_noref/``; ``car``
applies the CAR to them (equal to the ``cache125`` windows to 1e-5 µV).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.export import CLASS_MAP, N_VECTORS, lower, pte_operators, preprocess_params_v2, sha256
from src.quantize import (DEFAULT_TARGET, compile_spec, float_logits, input_qparams, int8_logits, load_float_model,
                          quantize, quantize_input, v2_run_config)

N_CH = 22
N_CALIB = 200
N_CALIB_TRIALS = 24
MI_CODES = (769, 770, 771, 772)
REPO_ROOT = Path(__file__).resolve().parents[1]
NOREF_CACHE_DIR = Path(os.environ.get("RB_CACHE125_NOREF_DIR", REPO_ROOT / "runs" / "cache125_noref"))
NOREF_CACHE_VERSION = 1


# ----------------------------------------------------------------------------- the fold
def car_matrix(n: int = N_CH) -> np.ndarray:
    return np.eye(n) - np.full((n, n), 1.0 / n)


def fold_matrix(A: np.ndarray, input_scale: float) -> np.ndarray:
    """``M_s = A_s C / input_scale`` (float64, 22 x 22)."""
    A = np.asarray(A, dtype=np.float64)
    return A @ car_matrix(A.shape[0]) / float(input_scale)


def fold_model(model: torch.nn.Module, M: np.ndarray) -> torch.nn.Module:
    """Copy of ``model`` whose input is ``X`` where ``model`` takes ``M X``: spatial weights
    ``w' = Mᵀ w`` and ``bn2.running_mean`` corrected for ``bn1``'s shift (see module doc)."""
    folded = copy.deepcopy(model).eval()
    M = torch.as_tensor(np.asarray(M), dtype=torch.float64)
    W = model.conv_spatial.weight.detach().double()  # (F1*D, 1, 22, 1)
    w = W[:, 0, :, 0]
    w_new = w @ M  # row k: w_kᵀ M = (Mᵀ w_k)ᵀ
    bn1 = model.bn1
    shift = (bn1.bias.double() - bn1.weight.double() * bn1.running_mean.double()
             / torch.sqrt(bn1.running_var.double() + bn1.eps))  # bn1(z) = a z + shift, per filter
    D = model.conv_spatial.out_channels // model.conv_spatial.in_channels
    group = torch.arange(w.shape[0]) // D
    delta = shift[group] * (w.sum(1) - w_new.sum(1))
    with torch.no_grad():
        folded.conv_spatial.weight.copy_(w_new[:, None, :, None].to(W.dtype))
        folded.bn2.running_mean.copy_((model.bn2.running_mean.double() - delta).to(torch.float32))
    return folded


def identity_check(model: torch.nn.Module, folded: torch.nn.Module, X_raw: np.ndarray, M: np.ndarray) -> float:
    """max |logit diff| (float32) between ``model(M X)`` and ``folded(X)`` on ``(n, 22, T)`` windows."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    mixed = np.einsum("cd,ndt->nct", np.asarray(M, dtype=np.float64), X_raw).astype(np.float32)
    a = float_logits(model, mixed[:, None])
    b = float_logits(folded, X_raw[:, None])
    return float(np.abs(a - b).max())


# ----------------------------------------------------------------------------- un-referenced 2a windows
def _build_noref(subject: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from src.data import cache125
    from src.data.download import load_raw
    from src.data.external import standardize_channels, use_upstream_provider
    from src.data.preprocess import UV_PER_V, design_filter

    use_upstream_provider()
    out = {}
    sos = design_filter(fs=cache125.FS_2A, **cache125.FILTER)
    for session, raw in sorted(load_raw(subject).items()):
        ev = np.asarray(raw.info["temp"]["events"], dtype=np.int64).copy()
        raw = standardize_channels(raw)
        xd = cache125.board_decimate(raw.get_data() * UV_PER_V, sos)  # no CAR: the board's new input
        ev[:, 0] //= cache125.DECIMATION
        keep = np.isin(ev[:, 2], MI_CODES)
        X = np.stack([xd[:, c + cache125.WINDOW_START:c + cache125.WINDOW_START + cache125.WINDOW]
                      for c in ev[keep, 0]]).astype(np.float32)
        out[int(session)] = (X, ev[keep, 2].astype(np.int64))
    return out


def noref_windows(subject: int, cache_dir: Path | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{session: (X (n, 22, 256) float32 µV, gdf_code (n,))}`` of all four MI classes, recording
    order, the ``cache125.windows_2a`` window without the CAR step. Left/right (codes 769/770) in
    order are the trials ``windows_2a`` returns."""
    path = Path(cache_dir or NOREF_CACHE_DIR) / f"BNCI2014_001_s{subject:02d}.npz"
    if path.exists():
        with np.load(path) as z:
            if int(z["version"]) == NOREF_CACHE_VERSION:
                return {int(s): (z["X"][z["session"] == s], z["code"][z["session"] == s]) for s in np.unique(z["session"])}
    data = _build_noref(subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    s = sorted(data)
    np.savez(tmp, X=np.concatenate([data[k][0] for k in s]), code=np.concatenate([data[k][1] for k in s]),
             session=np.concatenate([np.full(len(data[k][1]), k) for k in s]), version=np.int64(NOREF_CACHE_VERSION))
    os.replace(tmp, path)
    return data


def left_right(X: np.ndarray, code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(code, (769, 770))
    return X[keep], (code[keep] - 769).astype(np.int64)


def calibration_set(windows: dict, kind: str, seed: int = 0, n: int = N_CALIB) -> np.ndarray:
    """``(n, 1, 22, 256)`` un-referenced µV PTQ windows (labels / codes are never used)."""
    if kind == "session1":
        X = windows[1][0]
        idx = np.sort(np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False))
        X = X[idx]
    elif kind == "session2_calib":
        X = left_right(*windows[2])[0][:N_CALIB_TRIALS]
    else:
        raise ValueError(f"unknown calibration {kind!r}")
    return np.ascontiguousarray(X[:, None], dtype=np.float32)


# ----------------------------------------------------------------------------- params
def fused_params(run: dict, qparams: dict[int, tuple[float, int]], files: dict[int, dict], calib: str,
                 board_input: str = "car") -> dict:
    """``preprocess_params_fused.json``: the v2 schema with EA and the scale (and CAR) folded into
    the per-subject ``.pte`` and per-subject int8 input qparams. ``board_input`` ``"car"``: the
    qparams were calibrated on common-average-referenced windows, so the board keeps step 1."""
    p = preprocess_params_v2(run, {"ea": run["ea_matrices"]}, min(qparams), 1.0, 0)
    for key in ("subject", "ea_calibration"):
        p.pop(key)
    subjects = sorted(qparams)
    keep_car = board_input == "car"
    on_board = ["decode int16 -> uV", *(["1 CAR (subtract the mean of the 22, 250 Hz)"] if keep_car else []),
                "2 band-pass (sos, 250 Hz)", "3 decimate by 2", "4-5 ring buffer + window",
                "8 quantize with the subject's qparams", "9 run the subject's .pte", "10 argmax"]
    p.update({
        "version": "v2-fused",
        "subjects": subjects,
        "reference": "common_average_22" if keep_car else "folded",
        "reference_applied": ("subtract the per-sample mean of the 22 channels at 250 Hz, before the filter "
                              "(kept on the board: the int8 qparams are calibrated on referenced windows; the "
                              ".pte also contains the CAR, which is idempotent on them)") if keep_car else
                             "none on the board: the common average reference is folded into the .pte",
        "reference_folded_in_pte": True,
        "ea_matrix": "folded",
        "ea_matrices": "folded",
        "ea_calibration_folded": {"session": 2, "trials": f"1-{N_CALIB_TRIALS}", "labels_used": False,
                                  "eps": run["cfg"].get("ea_eps", run["params"].get("ea_eps"))},
        "input_scale": 1.0,
        "input_scale_folded": float(run["input_scale"]),
        "input_quant_scale": {str(s): float(qparams[s][0]) for s in subjects},
        "input_zero_point": {str(s): int(qparams[s][1]) for s in subjects},
        "input_formula": "x_int8 = clamp(round(x_uV_125hz_window / input_quant_scale[subject]) + "
                         "input_zero_point[subject], -128, 127); the window is "
                         + ("common-average-referenced, band-passed and decimated, not aligned or scaled" if keep_car
                            else "band-passed and decimated, not re-referenced, aligned or scaled"),
        "fused": {"inside_pte": ["6 EA (ea_matrices[subject])", "7 / input_scale", "1 CAR (I - 11^T/22)"],
                  "on_board": on_board,
                  "fold": "conv_spatial.weight w' = M_s^T w with M_s = ea_matrices[s] (I - 11^T/22) / "
                          "input_scale; bn2.running_mean -= bn1_shift * (sum(w) - sum(w'))",
                  "ptq_calibration": calib, "board_input": board_input},
        "pte": {str(s): files[s] for s in subjects},
        "class_map": CLASS_MAP,
    })
    return p


# ----------------------------------------------------------------------------- driver
def parse_subjects(spec: str) -> list[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return sorted({int(s) for s in spec.replace(" ", ",").split(",") if s})


def shipped_accuracy(run_dir: Path) -> dict[int, float]:
    m = json.loads((run_dir / "int8_metrics.json").read_text())
    return {int(s): float(v["int8"]["accuracy"]) for s, v in m["test"].items()}


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    run_dir, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = v2_run_config(run_dir)
    model = load_float_model(run_dir / "model.pt", cfg["window"], cfg["widths"])
    shape = tuple(cfg["input_shape"])
    spec = compile_spec(DEFAULT_TARGET)
    shipped = shipped_accuracy(run_dir)
    report: dict = {"run_dir": str(run_dir), "calibration": args.calib, "input": args.input, "subjects": {}}
    qparams, files = {}, {}
    for s in args.subjects:
        win = noref_windows(s)
        if args.input == "car":  # board keeps CAR: CAR'd windows (C is idempotent, the fold is unchanged)
            win = {k: (X - X.mean(axis=1, keepdims=True), c) for k, (X, c) in win.items()}
        X_test, y_test = left_right(*win[2])
        X_test, y_test = X_test[N_CALIB_TRIALS:], y_test[N_CALIB_TRIALS:]
        M = fold_matrix(cfg["ea_matrices"][s], cfg["input_scale"])
        folded = fold_model(model, M)
        ident = identity_check(model, folded, X_test[:100], M)
        rec = {"identity_max_abs_logit_diff_100": ident}
        print(f"s{s}: fold identity max |logit diff| on 100 windows = {ident:.3g}", flush=True)
        if ident > 1e-4:
            raise AssertionError(f"s{s}: fold identity {ident:.3g} > 1e-4")
        if args.check_only:
            report["subjects"][s] = rec
            continue
        calib = calibration_set(win, args.calib, args.seed)
        quantized = quantize(folded, calib, spec, input_shape=shape)
        scale, zp = input_qparams(quantized)
        x_in = X_test[:, None]
        q_test, f_test = int8_logits(quantized, x_in), float_logits(folded, x_in)
        x_q = quantize_input(x_in[:N_VECTORS], scale, zp)
        logits_vec = int8_logits(quantized, (x_q.astype(np.float32) - zp) * np.float32(scale))
        rec.update({
            "input_quant_scale": scale, "input_zero_point": zp, "n_calib": int(len(calib)),
            "int8_accuracy": float(np.mean(q_test.argmax(1) == y_test)),
            "float_accuracy": float(np.mean(f_test.argmax(1) == y_test)),
            "shipped_int8_accuracy": shipped[s],
            "input_clip_fraction": float(np.mean(np.abs(np.round(x_in / scale) + zp - np.clip(
                np.round(x_in / scale) + zp, -128, 127)) > 0)),
            "vectors_int8_accuracy": float(np.mean(logits_vec.argmax(1) == y_test[:N_VECTORS])),
        })
        rec["delta_points"] = 100 * (rec["int8_accuracy"] - shipped[s])
        print(f"s{s}: int8 {rec['int8_accuracy']:.3f} (float {rec['float_accuracy']:.3f}, shipped int8 "
              f"{shipped[s]:.3f}), qparams {scale:.6g} / {zp}, clip {rec['input_clip_fraction']:.2e}", flush=True)
        if args.no_pte:
            report["subjects"][s] = rec
            continue
        torch.save(folded.state_dict(), out / f"tinyeegnet_fused_s{s}_float.pt")
        np.savez(out / f"test_vectors_s{s}.npz", x=x_q, logits=logits_vec, y=y_test[:N_VECTORS].astype(np.int64))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            info, _, program = lower(quantized, spec, shape)
        (out / f"vela_s{s}.log").write_text(log.getvalue())
        pte_path = out / f"tinyeegnet_fused_s{s}_int8.pte"
        pte_path.write_bytes(bytes(program.buffer))
        files[s] = {"file": pte_path.name, "bytes": pte_path.stat().st_size, "sha256": sha256(pte_path)}
        rec.update({"pte": files[s], "ops_ethos_u": info.num_delegated_nodes, "ops_cpu": info.num_non_delegated_nodes,
                    "pte_cpu_operators": pte_operators(bytes(program.buffer))})
        qparams[s] = (scale, zp)
        report["subjects"][s] = rec
    if not args.check_only:
        acc = [r["int8_accuracy"] for r in report["subjects"].values()]
        report["int8_mean"] = float(np.mean(acc))
        report["shipped_int8_mean"] = float(np.mean([shipped[s] for s in report["subjects"]]))
        report["delta_mean_points"] = 100 * (report["int8_mean"] - report["shipped_int8_mean"])
        report["worst_delta_points"] = min(r["delta_points"] for r in report["subjects"].values())
        report["gate"] = bool(abs(report["delta_mean_points"]) <= 0.5 and report["worst_delta_points"] >= -2.0)
        print(f"mean int8 {report['int8_mean']:.4f} vs shipped {report['shipped_int8_mean']:.4f} "
              f"({report['delta_mean_points']:+.2f} pts, worst {report['worst_delta_points']:+.2f}), "
              f"gate {'PASS' if report['gate'] else 'FAIL'}", flush=True)
    if files:
        (out / "student.json").write_text(json.dumps(cfg["widths"], indent=1) + "\n")
        params = fused_params(cfg, qparams, files, args.calib, args.input)
        (out / "preprocess_params_fused.json").write_text(json.dumps(params, indent=1) + "\n")
    name = args.metrics_name or ("fused_metrics.json" if files else f"fused_metrics_{args.input}_{args.calib}_nopte.json")
    (out / name).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/v4soft/student_wide32_ft_seed0")
    p.add_argument("--subjects", default="1..9", type=parse_subjects, help="e.g. 1..9 or 3 or 1,3,5")
    p.add_argument("--out", default="artifacts/v4_fused/")
    p.add_argument("--calib", choices=["session1", "session2_calib"], default="session1")
    p.add_argument("--input", choices=["car", "noref"], default="car",
                   help="board input the qparams and vectors are made for: car = the board keeps CAR "
                        "(default, passes the accuracy gate); noref = un-referenced windows")
    p.add_argument("--seed", type=int, default=0, help="calibration draw")
    p.add_argument("--check-only", action="store_true", help="fold identity check only")
    p.add_argument("--no-pte", action="store_true", help="accuracy only, no lowering / files")
    p.add_argument("--metrics-name", default=None)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

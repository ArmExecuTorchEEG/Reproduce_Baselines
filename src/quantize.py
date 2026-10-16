# SPDX-License-Identifier: Apache-2.0
"""Post-training int8 quantization of TinyEEGNet for the Ethos-U85, and the host-side int8 check.

Mirrors ``CMSIS-Executorch/create_ai_layer.py:export_model``: ``EthosUQuantizer`` with the
per-channel symmetric config, ``prepare_pt2e`` -> calibrate -> ``convert_pt2e``. Calibration
uses 200 real aligned+scaled training windows (the same draw ``src.train`` saves as
``calibration.npz``), never target-subject data.

The delegated ``.pte`` does not run on the x86 runtime (the Ethos-U delegate needs the NPU),
so the host-side int8 numbers come from the quantized (fake-quant) graph module returned by
:func:`quantize`; the target check is the FVP run in Emulator_Deploy.

    python -m src.quantize --run runs/loso_s1_seed0

writes ``runs/loso_s1_seed0/int8_predictions.npz`` (``y_true``, ``logits``, ``test_trials``)
for the same test block as ``predictions.npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.tinyeegnet import INPUT_SHAPE, TinyEEGNet

N_CALIB = 200
DEFAULT_TARGET = "ethos-u85-256"


def compile_spec(target: str = DEFAULT_TARGET, system_config: str | None = None,
                 memory_mode: str | None = None, config_ini: str | None = None, extra_flags=None):
    """``EthosUCompileSpec`` as ``create_ai_layer.py:compile_spec`` builds it from the mlops node.

    Without ``config_ini`` the spec uses Vela's bundled ``Arm/vela.ini``; the Alif board
    configuration (``ensemble_vela.ini``, ``Ethos_U85_SRAM_MRAM`` / ``Shared_Sram``) only
    exists after ``cbuild setup`` in the starter, which re-exports the model itself.
    """
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    kwargs = {"target": target, "system_config": system_config, "memory_mode": memory_mode,
              "extra_flags": list(extra_flags or [])}
    if config_ini:
        kwargs["config_ini"] = config_ini
    return EthosUCompileSpec(**kwargs)


def load_float_model(checkpoint: str | Path) -> torch.nn.Module:
    """TinyEEGNet in eval mode with the weights of a ``model.pt`` state_dict."""
    model = TinyEEGNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def calibration_windows(data: dict, seed: int = 0, n: int = N_CALIB) -> np.ndarray:
    """``(n, 1, 22, 500)`` float32 aligned+scaled training windows from a ``loso_data`` /
    ``per_subject_data`` dict; the same seeded draw ``src.train`` writes to ``calibration.npz``."""
    from src.train import to_model_input

    X_train = to_model_input(data["train"][0], data["input_scale"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    return X_train[np.sort(idx)].numpy().astype(np.float32)


def quantize(model: torch.nn.Module, calib_windows, spec=None) -> torch.fx.GraphModule:
    """PTQ: export, annotate with ``EthosUQuantizer``, calibrate, convert. Returns the
    quantized (fake-quant, batch 1) GraphModule that ``src.export`` lowers to Ethos-U."""
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    spec = spec or compile_spec()
    samples = [torch.as_tensor(np.asarray(w, dtype=np.float32)).reshape(INPUT_SHAPE) for w in calib_windows]
    # check_guards=False: the default inserts a `_guards_fn` call_module, which the Arm
    # quantizer's DecomposeAvgPool2dPass cannot retrace ("call_module is not supported").
    graph = torch.export.export(model.eval(), (samples[0],)).module(check_guards=False)
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)
    return convert_pt2e(prepared)


def input_qparams(quantized: torch.fx.GraphModule) -> tuple[float, int]:
    """``(scale, zero_point)`` of the quantize op applied to the model input."""
    placeholder = next(n for n in quantized.graph.nodes if n.op == "placeholder")
    for user in placeholder.users:
        if "quantize_per_tensor" in str(user.target):
            scale, zero_point = user.args[1], user.args[2]
            return float(scale), int(zero_point)
    raise RuntimeError("model input is not followed by a quantize_per_tensor op")


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """float model input -> int8, as the quantize op does it (round half to even, clamp)."""
    q = np.round(np.asarray(x, dtype=np.float32) / np.float32(scale)) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


@torch.no_grad()
def int8_logits(quantized: torch.fx.GraphModule, X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Dequantized int8 logits ``(n, 2)`` float32 of the quantized model for ``(n, 1, 22, 500)``
    float inputs (the exported graph has a static batch of 1, so samples run one by one)."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    return np.concatenate([quantized(X[i : i + 1]).numpy() for i in range(len(X))]).astype(np.float32)


@torch.no_grad()
def float_logits(model: torch.nn.Module, X: np.ndarray | torch.Tensor) -> np.ndarray:
    return model.eval()(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float32)


def run_int8_predictions(run_dir: str | Path, checkpoint: str | Path | None = None) -> dict:
    """Quantize the run's checkpoint with its own training-data calibration windows and write
    ``int8_predictions.npz`` (dequantized int8 logits on the run's test block)."""
    from src.eval import compute_kpis
    from src.train import get_data, load_config, to_model_input

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metrics.json").read_text())
    if meta.get("model", "tinyeegnet") != "tinyeegnet":
        raise ValueError(f"{run_dir}: only TinyEEGNet runs are quantized, got {meta['model']}")
    cfg = load_config()
    data = get_data(meta["mode"], int(meta["subject"]), int(meta["seed"]), "session2",
                    cfg["training"]["val_fraction"])
    model = load_float_model(checkpoint or run_dir / "model.pt")
    quantized = quantize(model, calibration_windows(data, int(meta["seed"])))
    logits = int8_logits(quantized, to_model_input(data["test"][0], data["input_scale"]))
    y = data["test"][1]
    np.savez(run_dir / "int8_predictions.npz", y_true=y, logits=logits, test_trials=data["test_trials"])
    return compute_kpis(y, logits)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory with model.pt and metrics.json")
    p.add_argument("--checkpoint", default=None, help="override the run's model.pt")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    kpis = run_int8_predictions(args.run, args.checkpoint)
    print(f"{args.run} int8: " + " ".join(f"{k} {v:.4f}" for k, v in kpis.items()))


if __name__ == "__main__":
    main()

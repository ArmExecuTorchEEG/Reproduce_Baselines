# SPDX-License-Identifier: Apache-2.0
"""Train the v2 teacher ensemble (EEGNet-ELU, ShallowConvNet, DeepConvNet) at 125 Hz.

    python -m src.train_teacher --seed 0 --out runs/v2/teacher_seed0
    python -m src.train_teacher --seed 0 --out runs/v2/teacher_smoke --subset-cho 10 --epochs 2

Data: ``src.data.splits125.v2_data(seed, cfg)`` (Cho2017 + Lee2019 train/val, BCI IV 2a
session 2 calibration and test blocks; see that module). Every member is trained with the
v0.1 ``src.train.fit`` (Adam, cross-entropy, early stopping on the validation loss, best
weights restored) on the canonical crop ``X[..., crop_offset:crop_offset + window]`` of the
training windows, divided by ``input_scale``. No 2a trial is trained on, and the 2a test
block is only scored after training.

Writes to ``--out``:

- ``members/{eegnet,shallow,deep}.pt``  float state dicts
- ``teacher_logits.npz``  per member ``{m}_train``, ``{m}_val``, ``{m}_calib_{s}``, ``{m}_test_{s}``
  (logits, canonical window, no augmentation), plus ``y_train``, ``y_val``, ``y_test_{s}`` and
  ``subjects``
- ``metrics.json``  per member and for the ensemble (mean of the members' softmax): KPIs per
  2a subject (``src.eval.compute_kpis``), their mean over subjects, confusion matrices, fit info
- ``params.json``   input scale, per-subject EA matrices (``ea_matrices``), fs, window, crop
  offset, data meta and the full config echo
- ``history.json``  per-member epoch history; ``train.log``
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.special import softmax
from torch import nn

from src.eval import KPI_KEYS, compute_kpis, confusion
from src.models.convnets import DeepConvNet, ShallowConvNet
from src.models.eegnet import EEGNet
from src.train import CHANNELS, fit, predict, set_seed, to_model_input

ROOT = Path(__file__).resolve().parent.parent
CONFIG_V2 = ROOT / "configs" / "v2.yaml"
MEMBERS = ("eegnet", "shallow", "deep")
MEMBER_LABELS = {"eegnet": "EEGNet-ELU", "shallow": "ShallowConvNet", "deep": "DeepConvNet"}


# ----------------------------------------------------------------------------- config and data
def load_v2_config(path: str | Path = CONFIG_V2) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def subject_list(spec) -> list[int]:
    """Subjects from a config value: list of ints, one int, or an ``"a-b"`` range string."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        if "-" in spec:
            a, b = (int(v) for v in spec.split("-"))
            return list(range(a, b + 1))
        return [int(v) for v in spec.replace(",", " ").split()]
    return [int(v) for v in spec]


def subset_cho(cfg: dict, n: int) -> dict:
    """Copy of ``cfg`` keeping only the first ``n`` Cho2017 subjects (smoke runs)."""
    cfg = json.loads(json.dumps(cfg))
    cho = cfg["datasets"]["Cho2017"]
    exclude = set(subject_list(cho.get("exclude") or []))
    cho["subjects"] = [s for s in subject_list(cho["subjects"]) if s not in exclude][:n]
    return cfg


def load_v2_data(seed: int, cfg: dict) -> dict:
    """``src.data.splits125.v2_data`` (looked up at call time so tests can monkeypatch it)."""
    from src.data import splits125

    return splits125.v2_data(seed, cfg)


def canonical(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Canonical ``window``-sample crop at ``crop_offset`` of longer training windows;
    arrays that already have ``window`` samples are returned unchanged."""
    W, off = int(cfg["window"]), int(cfg["crop_offset"])
    if X.shape[-1] == W:
        return X
    return X[..., off : off + W]


def jsonable(obj):
    """Numpy scalars/arrays and int dict keys -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_params(out: Path, data: dict, cfg: dict, extra: dict | None = None) -> dict:
    """``params.json`` shared by teacher and student runs (and read by the export step)."""
    params = {
        "fs": float(cfg.get("fs_model", 125)),
        "n_channels": len(CHANNELS),
        "channels": CHANNELS,
        "window_samples": int(cfg["window"]),
        "train_window_samples": int(cfg.get("train_window", cfg["window"])),
        "crop_offset": int(cfg["crop_offset"]),
        "input_shape": [1, 1, len(CHANNELS), int(cfg["window"])],
        "units": "uV",
        "reference": "common_average_22",
        "filter": cfg.get("filter"),
        "ea_eps": cfg.get("ea_eps"),
        "ea_matrices": {str(s): np.asarray(A).tolist() for s, A in data["ea"].items()},
        "input_scale": float(data["input_scale"]),
        "class_map": {"0": "left_hand (769)", "1": "right_hand (770)"},
        "meta": jsonable(data.get("meta", {})),
        "config": jsonable(cfg),
    }
    params.update(jsonable(extra or {}))
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=1)
    return params


# ----------------------------------------------------------------------------- models
def build_member(name: str, cfg: dict) -> nn.Module:
    n_samples = int(cfg["window"])
    if name == "eegnet":
        return EEGNet(n_samples=n_samples, kern_t=64, dropout=cfg["training"]["dropout"])
    if name == "shallow":
        return ShallowConvNet(n_samples=n_samples)
    if name == "deep":
        return DeepConvNet(n_samples=n_samples)
    raise ValueError(name)


def load_members(teacher_dir: str | Path, cfg: dict, device: str = "cpu") -> dict[str, nn.Module]:
    """Trained teacher members in eval mode, gradients off."""
    members = {}
    for name in MEMBERS:
        model = build_member(name, cfg)
        state = torch.load(Path(teacher_dir) / "members" / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        members[name] = model
    return members


def ensemble_logits(logits: list[np.ndarray]) -> np.ndarray:
    """log of the mean member softmax; ``softmax`` of it is the ensemble probability."""
    p = np.mean([softmax(np.asarray(z, dtype=np.float64), axis=1) for z in logits], axis=0)
    return np.log(np.clip(p, 1e-12, None)).astype(np.float32)


def subject_report(test: dict, logits: dict) -> dict:
    """KPIs and confusion per 2a subject plus the mean of each KPI over subjects."""
    per = {s: compute_kpis(test[s][1], logits[s]) for s in test}
    return {"test": {str(s): k for s, k in per.items()},
            "test_mean": {k: float(np.mean([per[s][k] for s in per])) for k in KPI_KEYS},
            "confusion": {str(s): confusion(test[s][1], logits[s]).tolist() for s in test}}


# ----------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict:
    cfg = load_v2_config(args.config)
    if args.subset_cho:
        cfg = subset_cho(cfg, args.subset_cho)
    tr = cfg["training"]
    lr = args.lr if args.lr is not None else tr["lr"]
    batch_size = args.batch_size or tr["batch_size"]
    max_epochs = args.epochs or tr["max_epochs"]
    patience = args.patience or tr["patience"]
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "members").mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    t_all = time.time()
    set_seed(args.seed)
    data = load_v2_data(args.seed, cfg)
    scale = float(data["input_scale"])
    subjects = sorted(data["test"])
    X_train = to_model_input(canonical(data["train"][0], cfg), scale)
    X_val = to_model_input(canonical(data["val"][0], cfg), scale)
    X_calib = {s: to_model_input(data["calib"][s], scale) for s in subjects}
    X_test = {s: to_model_input(data["test"][s][0], scale) for s in subjects}
    y_train, y_val = data["train"][1], data["val"][1]
    log(f"teacher seed {args.seed}: train {tuple(X_train.shape)} val {tuple(X_val.shape)} test subjects {subjects} "
        f"({sum(len(X_test[s]) for s in subjects)} trials) input_scale {scale:.4f} threads {args.threads} "
        f"device {args.device} meta {json.dumps(jsonable(data.get('meta', {})))}")

    arrays: dict[str, np.ndarray] = {"y_train": y_train, "y_val": y_val, "subjects": np.array(subjects)}
    for s in subjects:
        arrays[f"y_test_{s}"] = data["test"][s][1]
    metrics: dict = {"seed": args.seed, "n_train": int(len(X_train)), "n_val": int(len(X_val)),
                     "n_calib": int(sum(len(X_calib[s]) for s in subjects)),
                     "n_test": int(sum(len(X_test[s]) for s in subjects)), "input_scale": scale,
                     "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
                     "threads": args.threads, "device": args.device, "meta": jsonable(data.get("meta", {})),
                     "members": {}}
    histories = {}
    test_logits = {name: {} for name in MEMBERS}
    val_logits = {}
    for name in MEMBERS:
        set_seed(args.seed)
        model = build_member(name, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"--- {MEMBER_LABELS[name]} ({n_params} params)")
        info = fit(model, X_train, y_train, X_val, y_val, lr=lr, batch_size=batch_size, max_epochs=max_epochs,
                   patience=patience, seed=args.seed, device=args.device, log=log)
        model.cpu().eval()
        torch.save(model.state_dict(), out / "members" / f"{name}.pt")
        arrays[f"{name}_train"] = predict(model, X_train)
        arrays[f"{name}_val"] = val_logits[name] = predict(model, X_val)
        for s in subjects:
            arrays[f"{name}_calib_{s}"] = predict(model, X_calib[s])
            arrays[f"{name}_test_{s}"] = test_logits[name][s] = predict(model, X_test[s])
        rep = subject_report(data["test"], test_logits[name])
        histories[name] = info.pop("history")
        metrics["members"][name] = {"label": MEMBER_LABELS[name], "params": n_params, **info, **rep}
        log(f"{MEMBER_LABELS[name]} test mean: " + " ".join(f"{k} {v:.4f}" for k, v in rep["test_mean"].items()))

    ens_test = {s: ensemble_logits([test_logits[m][s] for m in MEMBERS]) for s in subjects}
    ens_val = ensemble_logits([val_logits[m] for m in MEMBERS])
    yv = torch.as_tensor(y_val, dtype=torch.long)
    metrics["ensemble"] = {
        "label": "Teacher ensemble", "members": list(MEMBERS),
        "params": sum(metrics["members"][m]["params"] for m in MEMBERS),
        "val_loss": float(nn.functional.nll_loss(torch.from_numpy(ens_val), yv)),
        "val_acc": float(np.mean(ens_val.argmax(1) == y_val)),
        **subject_report(data["test"], ens_test),
    }
    np.savez(out / "teacher_logits.npz", **arrays)
    write_params(out, data, cfg, {"teacher_members": list(MEMBERS), "seed": args.seed})
    metrics["wall_time_s"] = time.time() - t_all
    metrics["versions"] = {"torch": torch.__version__, "python": platform.python_version()}
    with open(out / "metrics.json", "w") as f:
        json.dump(jsonable(metrics), f, indent=1)
    with open(out / "history.json", "w") as f:
        json.dump(histories, f)
    log("ensemble test mean: " + " ".join(f"{k} {v:.4f}" for k, v in metrics["ensemble"]["test_mean"].items())
        + f" | wall {metrics['wall_time_s']:.0f}s")
    log_f.close()
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--subset-cho", type=int, default=None, help="keep only the first N Cho2017 subjects (smoke)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

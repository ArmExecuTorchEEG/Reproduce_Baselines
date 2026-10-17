"""v2 data split: train on Cho2017 + Lee2019_MI, test on all 9 BCI IV 2a subjects.

``v2_data(seed, cfg)`` returns::

    {"train": (X (n, 22, 306) float32, y int64),   # EA per training subject-session, not scaled
     "val":   (X (m, 22, 306) float32, y int64),   # stratified 10 % of every training subject-session
     "calib": {subject: X (24, 22, 256) float32},  # 2a session 2 trials 1-24, EA applied, no labels
     "test":  {subject: (X (120, 22, 256), y)},    # 2a session 2 trials 25-144, EA applied
     "ea":    {subject: A (22, 22) float64},       # 2a EA matrix from the 24 calibration trials
     "input_scale": float,                         # std of the aligned training X
     "meta": {...}}

Model input = ``X[:, None] / input_scale``. The canonical window of a training trial is
``X[..., 25:281]``; test and calib arrays are already that window. No 2a trial is used for
training or validation; 2a session 1 is unused. EA uses ``align.ea_matrix`` with
``cfg["ea_eps"]`` (default 1e-3): common average reference leaves the 22x22 covariance with
one (numerically) zero eigenvalue, which the ``eps * trace / 22`` ridge lifts.

Validation draws are seeded per ``(seed, dataset, subject, session)`` so a subject's split
does not change when other subjects are added or skipped. Subjects whose raw files are not
downloaded yet are skipped with a warning and listed in ``meta["skipped"]``.
"""

from __future__ import annotations

import warnings
import zlib
from pathlib import Path

import numpy as np
import yaml

from src.data import cache125
from src.data.align import apply_ea, ea_matrix
from src.data.external import DataNotDownloaded
from src.data.splits import stratified_holdout

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "v2.yaml"
DEFAULT_EA_EPS = 1e-3


def load_config(path: Path = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_subjects(spec) -> list[int]:
    """``[1, 2, 3]``, ``5`` or ``"1-52"`` -> sorted list of ints."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            a, _, b = part.strip().partition("-")
            out.extend(range(int(a), int(b or a) + 1))
        return sorted(set(out))
    return sorted({int(s) for s in spec})


def training_subjects(cfg: dict) -> list[tuple[str, int]]:
    """``[(dataset, subject), ...]`` from ``cfg["datasets"]`` minus each ``exclude`` list."""
    pairs = []
    for dataset, spec in cfg["datasets"].items():
        excl = set(expand_subjects(spec.get("exclude") or []))
        pairs += [(dataset, s) for s in expand_subjects(spec["subjects"]) if s not in excl]
    return pairs


def check_config(cfg: dict) -> None:
    """The cached windows are built with the constants of ``cache125``; refuse a config
    that silently disagrees with them."""
    want = {"fs_model": cache125.FS_MODEL, "window": cache125.WINDOW, "train_window": cache125.TRAIN_WINDOW,
            "crop_offset": cache125.CROP_OFFSET, "window_start": cache125.WINDOW_START}
    for key, value in want.items():
        if key in cfg and float(cfg[key]) != float(value):
            raise ValueError(f"cfg[{key!r}] = {cfg[key]} but cache125 builds {value}")
    for key, value in cache125.FILTER.items():
        if float(cfg.get("filter", {}).get(key, value)) != float(value):
            raise ValueError(f"cfg['filter'][{key!r}] = {cfg['filter'][key]} but cache125 uses {value}")


def _rng(seed: int, dataset: str, subject: int, session: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(dataset.encode()), int(subject), int(session)])


def v2_data(seed: int, cfg: dict | None = None, loader=None) -> dict:
    """Training/validation from the external datasets, calibration/test from 2a (see module doc).

    ``loader(dataset, subject) -> {session: (X, y)}`` defaults to
    :func:`src.data.cache125.cached_windows`; it may raise ``DataNotDownloaded``.
    """
    cfg = load_config() if cfg is None else cfg
    check_config(cfg)
    loader = loader or cache125.cached_windows
    eps = float(cfg.get("ea_eps", DEFAULT_EA_EPS))
    val_fraction = float(cfg.get("training", {}).get("val_fraction", 0.1))
    test_cfg = cfg.get("test", {})
    test_dataset = test_cfg.get("dataset", cache125.TEST_DATASET)
    test_session = int(test_cfg.get("session", 2))
    n_calib = int(test_cfg.get("calib_trials", 24))

    Xtr, ytr, Xva, yva = [], [], [], []
    sources: dict[str, dict] = {}
    skipped: list[str] = []
    n_sessions = 0
    for dataset, subject in training_subjects(cfg):
        if dataset == test_dataset:
            raise ValueError("the test dataset must not be a training source")
        src = sources.setdefault(dataset, {"subjects": 0, "subject_sessions": 0, "train": 0, "val": 0,
                                           "skipped": 0})
        try:
            data = loader(dataset, subject)
        except DataNotDownloaded as exc:
            warnings.warn(f"skipping {dataset} subject {subject}: not downloaded ({exc})", stacklevel=2)
            src["skipped"] += 1
            skipped.append(f"{dataset}_s{subject:02d}")
            continue
        src["subjects"] += 1
        for session in sorted(data):
            X, y = data[session]
            Xa = apply_ea(np.asarray(X, dtype=np.float32), ea_matrix(X, eps))
            tr, va = stratified_holdout(y, val_fraction, _rng(seed, dataset, subject, session))
            Xtr.append(Xa[tr]); ytr.append(y[tr])
            Xva.append(Xa[va]); yva.append(y[va])
            src["subject_sessions"] += 1
            src["train"] += len(tr)
            src["val"] += len(va)
            n_sessions += 1
    if not Xtr:
        raise RuntimeError("no training subject could be loaded")
    X_train = np.concatenate(Xtr).astype(np.float32, copy=False)
    y_train = np.concatenate(ytr).astype(np.int64)
    X_val = np.concatenate(Xva).astype(np.float32, copy=False)
    y_val = np.concatenate(yva).astype(np.int64)

    calib, test, ea = {}, {}, {}
    for subject in expand_subjects(test_cfg.get("subjects", list(range(1, 10)))):
        X2, y2 = loader(test_dataset, subject)[test_session]
        if len(y2) <= n_calib:
            raise ValueError(f"{test_dataset} s{subject}: only {len(y2)} trials")
        A = ea_matrix(X2[:n_calib], eps)
        calib[subject] = apply_ea(np.asarray(X2[:n_calib], dtype=np.float32), A)
        test[subject] = (apply_ea(np.asarray(X2[n_calib:], dtype=np.float32), A),
                         np.asarray(y2[n_calib:], dtype=np.int64))
        ea[subject] = np.asarray(A, dtype=np.float64)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "calib": calib,
        "test": test,
        "ea": ea,
        "input_scale": float(np.std(X_train)),
        "meta": {
            "n_train_subject_sessions": n_sessions,
            "sources": sources,
            "skipped": skipped,
            "fs": int(cache125.FS_MODEL),
            "window": cache125.WINDOW,
            "train_window": cache125.TRAIN_WINDOW,
            "crop_offset": cache125.CROP_OFFSET,
            "window_start": cache125.WINDOW_START,
            "ea_eps": eps,
            "seed": int(seed),
        },
    }


__all__ = ["v2_data", "load_config", "expand_subjects", "training_subjects", "check_config"]

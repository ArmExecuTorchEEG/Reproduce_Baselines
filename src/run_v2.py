# SPDX-License-Identifier: Apache-2.0
"""Run the v2 teacher/student experiments and write ``results/results_v2.md``.

    python -m src.run_v2 --jobs 3 --seeds 0 1 2 --students wide32 control
    python -m src.run_v2 --skip-train                      # only rebuild the report from runs/v2

Runs (one directory each under ``runs/v2/``, finished runs are skipped):
  teacher_seed{K}             src.train_teacher (EEGNet-ELU, ShallowConvNet, DeepConvNet)
  student_{name}_seed{K}      src.distill from teacher_seed{K}

Every training process gets ``--threads`` torch threads (1 by default, as ``src.run_all``);
teachers run first, then the students. Int8 rows are read from
``runs/v2/student_{name}_seed{K}/int8_predictions.npz`` (same keys as ``predictions.npz``:
``subjects``, ``y_true_{s}``, ``logits_{s}``) when present, otherwise marked pending.
Test set for every row: BCI IV 2a session 2 trials 25-144 of subjects 1-9, never trained on.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import torch

from src.eval import KPI_KEYS, compute_kpis, confusion
from src.run_all import across_subjects, fmt, fmt_pm, git_commit, subject_mean, subject_std
from src.train_teacher import CONFIG_V2, MEMBER_LABELS, MEMBERS, build_member, load_v2_config, subject_list

ROOT = Path(__file__).resolve().parent.parent
RUNS_V2 = ROOT / "runs" / "v2"
RESULTS = ROOT / "results"


def teacher_dir(runs: Path, seed: int) -> Path:
    return runs / f"teacher_seed{seed}"


def student_dir(runs: Path, name: str, seed: int) -> Path:
    return runs / f"student_{name}_seed{seed}"


# ----------------------------------------------------------------------------- training
def run_job(cmd: list[str], out: Path) -> tuple[int, float]:
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(out / "stdout.log", "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t


def common_flags(a: argparse.Namespace) -> list[str]:
    flags = ["--config", str(a.config), "--threads", str(a.threads), "--device", a.device]
    if a.epochs:
        flags += ["--epochs", str(a.epochs)]
    if a.subset_cho:
        flags += ["--subset-cho", str(a.subset_cho)]
    return flags


def train_all(a: argparse.Namespace) -> None:
    runs = Path(a.runs_dir)
    py = sys.executable
    teachers = [k for k in a.seeds if not (teacher_dir(runs, k) / "metrics.json").exists()]
    students = [(n, k) for n in a.students for k in a.seeds if not (student_dir(runs, n, k) / "metrics.json").exists()]
    print(f"{len(teachers)} teacher and {len(students)} student runs pending, {a.jobs} in parallel x {a.threads} "
          f"threads", flush=True)
    if teachers or students:
        # Build every per-subject cache once before parallel workers start.
        from src.train_teacher import load_v2_data, subset_cho

        cfg = load_v2_config(a.config)
        load_v2_data(a.seeds[0], subset_cho(cfg, a.subset_cho) if a.subset_cho else cfg)
    t0 = time.time()

    def teacher(k: int):
        out = teacher_dir(runs, k)
        return ("teacher", k), *run_job([py, "-m", "src.train_teacher", "--seed", str(k), "--out", str(out),
                                         *common_flags(a)], out)

    def student(job):
        n, k = job
        out = student_dir(runs, n, k)
        if not (teacher_dir(runs, k) / "metrics.json").exists():
            return (n, k), -1, 0.0
        return (n, k), *run_job([py, "-m", "src.distill", "--seed", str(k), "--student", n, "--teacher",
                                 str(teacher_dir(runs, k)), "--out", str(out), *common_flags(a)], out)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for job, rc, dt in ex.map(teacher, teachers):
            print(f"teacher seed{job[1]}: rc {rc} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
        for job, rc, dt in ex.map(student, students):
            msg = "teacher missing" if rc == -1 else f"rc {rc}"
            print(f"student {job[0]} seed{job[1]}: {msg} ({dt:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- collection
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def per_subject_npz(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """``{subject: (y_true, logits)}`` from a ``predictions.npz`` / ``int8_predictions.npz``."""
    if not path.exists():
        return None
    z = np.load(path)
    return {int(s): (z[f"y_true_{int(s)}"], z[f"logits_{int(s)}"]) for s in z["subjects"]}


def _cell() -> dict:
    return {"float": [], "int8": [], "confusion": [], "confusion_int8": [], "seeds": []}


def collect(runs: Path, seeds, students, subjects) -> tuple[dict, dict]:
    """``table[tag][subject] = {"float": [kpis per seed], "int8": [...], "confusion": [...]}`` and run facts."""
    tags = [*MEMBERS, "ensemble", *(f"student_{n}" for n in students)]
    table = {t: {s: _cell() for s in subjects} for t in tags}
    facts: dict = {"teacher": {}, "student": {}}
    for k in seeds:
        m = load_json(teacher_dir(runs, k) / "metrics.json")
        if m is None:
            continue
        facts["teacher"][k] = m
        for tag in (*MEMBERS, "ensemble"):
            src = m["ensemble"] if tag == "ensemble" else m["members"][tag]
            for s in subjects:
                if str(s) in src["test"]:
                    c = table[tag][s]
                    c["float"].append(src["test"][str(s)]); c["confusion"].append(np.array(src["confusion"][str(s)]))
                    c["seeds"].append(k)
    for n in students:
        tag = f"student_{n}"
        for k in seeds:
            d = student_dir(runs, n, k)
            m = load_json(d / "metrics.json")
            if m is None:
                continue
            facts["student"][(n, k)] = m
            q = per_subject_npz(d / "int8_predictions.npz")
            for s in subjects:
                c = table[tag][s]
                if str(s) in m["test"]:
                    c["float"].append(m["test"][str(s)]); c["confusion"].append(np.array(m["confusion"][str(s)]))
                    c["seeds"].append(k)
                if q is not None and s in q:
                    c["int8"].append(compute_kpis(*q[s])); c["confusion_int8"].append(confusion(*q[s]))
    return table, facts


# ----------------------------------------------------------------------------- params and MACs
def count_macs(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulates of conv + linear layers for one window (as ``src.export.count_macs``)."""
    try:
        from src.export import count_macs as export_count_macs

        if "input_shape" in inspect.signature(export_count_macs).parameters:
            return int(export_count_macs(model, input_shape=input_shape)[0])
    except Exception:
        pass
    total = 0

    def hook(mod, inp, out):
        nonlocal total
        if isinstance(mod, torch.nn.Conv2d):
            total += out.numel() * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        else:
            total += mod.in_features * mod.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return int(total)


def model_costs(cfg: dict, students) -> dict[str, tuple[int, int]]:
    """``tag -> (params, MACs)`` for one ``(1, 1, 22, window)`` window."""
    from src.distill import build_student

    shape = (1, 1, 22, int(cfg["window"]))
    costs = {}
    for name in MEMBERS:
        m = build_member(name, cfg)
        costs[name] = (sum(p.numel() for p in m.parameters()), count_macs(m, shape))
    costs["ensemble"] = tuple(sum(costs[n][i] for n in MEMBERS) for i in (0, 1))
    for n in students:
        m = build_student(n, cfg)
        costs[f"student_{n}"] = (sum(p.numel() for p in m.parameters()), count_macs(m, shape))
    return costs


V01_LABEL = "*v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)*"


def v01_row() -> dict | None:
    """v0.1 LOSO TinyEEGNet (250 Hz, 500 samples, trained on 8 of the 2a subjects) from results/summary.json."""
    s = load_json(RESULTS / "summary.json")
    if not s or "loso" not in s.get("rows", {}):
        return None
    rows = s["rows"]["loso"]
    out = {}
    for prec in ("float", "int8"):
        per = {int(k): v[prec] for k, v in rows.items() if v.get(prec)}
        out[prec] = per
    return out


# ----------------------------------------------------------------------------- report
def summary_rows(students, table) -> list[tuple[str, str, str]]:
    """``(label, tag, precision)`` rows of the summary table."""
    rows = [(MEMBER_LABELS[m], m, "float") for m in MEMBERS]
    rows.append(("Teacher ensemble (mean softmax)", "ensemble", "float"))
    for n in students:
        tag = f"student_{n}"
        rows.append((f"Student {n}", tag, "float"))
        if any(table[tag][s]["int8"] for s in table[tag]):
            rows.append((f"Student {n}", tag, "int8"))
    return rows


def select_student(table, students, subjects, costs) -> tuple[str | None, str]:
    """Highest mean int8 accuracy over subjects, tie-break on MACs; float when no int8 exists."""
    for prec in ("int8", "float"):
        cand = []
        for n in students:
            m, _, cnt = across_subjects(table, f"student_{n}", prec, subjects, "accuracy")
            if cnt:
                cand.append((-round(m, 6), costs[f"student_{n}"][1], n))
        if cand:
            return sorted(cand)[0][2], prec
    return None, ""


def export_seed(facts, name: str, seeds) -> int | None:
    """Seed of the student run with the lowest best validation KL to the teacher (``best_val_kl``,
    the distillation selection criterion; never the test set)."""
    kls = {k: facts["student"][(name, k)]["best_val_kl"] for k in seeds if (name, k) in facts["student"]}
    return min(kls, key=kls.get) if kls else None


def write_report(table, facts, cfg, seeds, students, subjects, costs, export_subject: int = 3,
                 runs: Path = RUNS_V2) -> str:
    L = []
    runs = Path(runs)
    runs_shown = runs.relative_to(ROOT) if runs.is_relative_to(ROOT) else runs
    W, off = int(cfg["window"]), int(cfg["crop_offset"])
    fs = cfg.get("fs_model", 125)
    start = cfg.get("window_start", 0.5 * fs) / fs  # canonical window start after the cue, s
    L.append("# Reproduce_Baselines v2 results: teacher ensemble distilled into TinyEEGNet at 125 Hz\n")
    L.append("## 1. Setup\n")
    meta = next(iter(facts["teacher"].values()), {}).get("meta", {}) if facts["teacher"] else {}
    L.append(f"Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, "
             f"common average reference over the 22, {fs} Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per "
             f"subject-session, one global `input_scale`. Training windows {cfg.get('train_window', W)} samples with "
             f"random {W}-sample crops for the students; teachers train on the canonical crop at offset {off} "
             f"(cue + {start:.3f} s to cue + {start + W / fs:.3f} s). Validation = stratified 10 % of every training "
             "subject-session (early stopping only). **Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects "
             f"{subjects[0]}–{subjects[-1]}, session 2 trials 25–144 (120 per subject), never trained on.** "
             "Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream "
             "with teacher soft labels only (their labels are never used). "
             f"Seeds {list(seeds)}; each subject's value is the mean over seeds, ± is the std across subjects. "
             "KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.\n")
    kd = cfg.get("distill", {})
    L.append(f"Distillation: loss = {1 - kd.get('alpha', 0):.2g} CE + {kd.get('alpha', 0)} T² KL, T = "
             f"{kd.get('temperature')}, noise std {kd.get('noise_std')}, channel dropout {kd.get('channel_dropout')}; "
             "the teacher scores each augmented batch on the fly.\n")
    try:
        vers = {p: pkg_version(p) for p in ("torch", "moabb", "mne", "scikit-learn", "scipy", "numpy")}
    except Exception:
        vers = {"torch": torch.__version__}
    L.append("Environment: Python " + platform.python_version() + ", " + ", ".join(f"{k} {v}" for k, v in vers.items())
             + f"; git commit `{git_commit()}`; generated {time.strftime('%Y-%m-%d %H:%M')}.\n")

    # summary
    rows = summary_rows(students, table)
    L.append(f"## 2. Summary (mean ± std across the {len(subjects)} 2a subjects)\n")
    L.append("| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, tag, prec in rows:
        cells = [fmt_pm(*across_subjects(table, tag, prec, subjects, k)[:2], pct=(k == "accuracy")) for k in KPI_KEYS]
        n = across_subjects(table, tag, prec, subjects, "accuracy")[2]
        n_seeds = max((len(table[tag][s][prec]) for s in subjects), default=0)
        p, macs = costs[tag]
        L.append(f"| {label} | {prec} | " + " | ".join(cells) + f" | {p:,} | {macs / 1e6:.2f} M | {n} | {n_seeds} |")
    v01 = v01_row()
    if v01:
        from src.models.tinyeegnet import TinyEEGNet

        ref = TinyEEGNet()
        p, macs = sum(q.numel() for q in ref.parameters()), count_macs(ref, (1, 1, 22, 500))
        for prec in ("float", "int8"):
            per = v01[prec]
            if not per:
                continue
            cells = []
            for k in KPI_KEYS:
                vals = [per[s][k] for s in per]
                cells.append(fmt_pm(float(np.mean(vals)), float(np.std(vals)), pct=(k == "accuracy")))
            L.append(f"| {V01_LABEL} | {prec} | " + " | ".join(cells)
                     + f" | {p:,} | {macs / 1e6:.2f} M | {len(per)} | 3 |")
        L.append("\nThe v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with "
                 "500-sample windows; it is shown for context only (`results/results.md`).\n")
    else:
        L.append("")
    L.append("MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. "
             "Ensemble params/MACs are the sum of its three members (offline only).\n")

    # per-subject accuracy
    L.append("## 3. Per-subject accuracy (%)\n")
    L.append("| Model | Precision | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean |")
    L.append("|---|---|" + "---|" * (len(subjects) + 1))
    for label, tag, prec in rows:
        cells = [fmt(subject_mean(table[tag][s][prec], "accuracy"), True) for s in subjects]
        L.append(f"| {label} | {prec} | " + " | ".join(cells)
                 + f" | {fmt(across_subjects(table, tag, prec, subjects, 'accuracy')[0], True)} |")
    for prec in ("float", "int8"):
        per = v01[prec] if v01 else {}
        if not per:
            continue
        cells = [fmt(per[s]["accuracy"], True) if s in per else "n/a" for s in subjects]
        mean = fmt(float(np.mean([per[s]["accuracy"] for s in subjects if s in per])), True)
        L.append(f"| {V01_LABEL} | {prec} | " + " | ".join(cells) + f" | {mean} |")
    if v01:
        L.append("\nv0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.")
    L.append("")

    # shipped student
    chosen, chosen_prec = select_student(table, students, subjects, costs)
    L.append("## 4. Shipped student\n")
    if chosen is None:
        L.append("pending (no finished student runs).\n")
    else:
        tag = f"student_{chosen}"
        k_exp = export_seed(facts, chosen, seeds)
        L.append(f"Selected: **student {chosen}** (highest mean {chosen_prec} accuracy across subjects, tie-break on "
                 f"MACs{'; int8 results pending, selected on float' if chosen_prec == 'float' else ''}). Export subject "
                 f"A0{export_subject}, seed {k_exp} (lowest best validation KL to the teacher: "
                 + (f"{facts['student'][(chosen, k_exp)]['best_val_kl']:.4f}" if k_exp is not None else "n/a")
                 + f"), checkpoint `{runs_shown}/student_{chosen}_seed{k_exp}/model.pt`.\n")
        L.append("Per-subject KPIs, float (mean ± std over seeds):\n")
        L.append("| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |")
        L.append("|---|---|---|---|---|---|---|")
        epochs = ", ".join(str(facts["student"][(chosen, k)]["best_epoch"]) for k in seeds if (chosen, k) in facts["student"])
        for s in subjects:
            c = table[tag][s]["float"]
            cells = [fmt_pm(subject_mean(c, k), subject_std(c, k), pct=(k == "accuracy")) for k in KPI_KEYS]
            L.append(f"| A0{s}{' (export)' if s == export_subject else ''} | " + " | ".join(cells) + f" | {epochs} |")
        cells = [fmt_pm(*across_subjects(table, tag, "float", subjects, k)[:2], pct=(k == "accuracy")) for k in KPI_KEYS]
        L.append("| **mean ± std** | " + " | ".join(cells) + " | |\n")

    # int8 minus float
    L.append("## 5. Int8 minus float accuracy (points)\n")
    have_int8 = [n for n in students if any(table[f"student_{n}"][s]["int8"] for s in subjects)]
    if not have_int8:
        L.append("pending (no `int8_predictions.npz` in the student run directories yet).\n")
    else:
        L.append("| Student | " + " | ".join(f"A0{s}" for s in subjects) + " | Mean |")
        L.append("|---|" + "---|" * (len(subjects) + 1))
        for n in have_int8:
            c = table[f"student_{n}"]
            deltas = [subject_mean(c[s]["int8"], "accuracy") - subject_mean(c[s]["float"], "accuracy") for s in subjects]
            cells = ["pending" if d != d else f"{100 * d:+.1f}" for d in deltas]
            valid = [d for d in deltas if d == d]
            mean = f"{100 * np.mean(valid):+.1f}" if valid else "pending"
            L.append(f"| {n} | " + " | ".join(cells) + f" | {mean} |")
        L.append("")

    # training facts
    L.append("## 6. Training facts\n")
    if facts["teacher"]:
        m0 = next(iter(facts["teacher"].values()))
        L.append(f"Trials: train {m0['n_train']:,}, validation {m0['n_val']:,}, 2a calibration {m0['n_calib']:,} "
                 f"(unlabelled), 2a test {m0['n_test']:,}. Data meta: `{json.dumps(meta)}`.\n")
        L.append("Teachers stop and select on the validation cross-entropy, students on the validation KL to the "
                 "teacher ensemble (the combined distillation loss at that epoch in brackets).\n")
        L.append("| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |")
        L.append("|---|---|---|---|---|")
        for k, m in sorted(facts["teacher"].items()):
            be = ", ".join(f"{MEMBER_LABELS[n]} {m['members'][n]['best_epoch']}" for n in MEMBERS)
            vl = ", ".join(f"{m['members'][n]['best_val_loss']:.4f}" for n in MEMBERS)
            L.append(f"| teacher | {k} | {be} | {vl} | {m['wall_time_s']:.0f} |")
        for (n, k), m in sorted(facts["student"].items()):
            L.append(f"| student {n} | {k} | {m['best_epoch']} | {m['best_val_kl']:.4f} ({m['best_val_loss']:.4f}) "
                     f"| {m['wall_time_s']:.0f} |")
        L.append("")
    else:
        L.append("pending (no finished teacher runs).\n")
    L.append(f"Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): "
             f"`results/confusion_v2/A01.md` … `A0{subjects[-1]}.md`.\n")
    return "\n".join(L)


def write_confusions(table, students, subjects, results: Path) -> None:
    d = results / "confusion_v2"
    d.mkdir(parents=True, exist_ok=True)
    tags = [(MEMBER_LABELS[m], m) for m in MEMBERS] + [("Teacher ensemble", "ensemble")] + \
           [(f"Student {n}", f"student_{n}") for n in students]
    for s in subjects:
        L = [f"# Confusion matrices v2, subject A0{s}\n",
             "Rows = true (left, right), columns = predicted (left, right); summed over seeds.\n"]
        for label, tag in tags:
            for key, prec in (("confusion", "float"), ("confusion_int8", "int8")):
                cms = table[tag][s][key]
                if not cms:
                    continue
                cm = np.sum(cms, axis=0)
                L.append(f"## {label}, {prec}\n\n| | pred left | pred right |\n|---|---|---|\n"
                         f"| true left | {cm[0, 0]} | {cm[0, 1]} |\n| true right | {cm[1, 0]} | {cm[1, 1]} |\n")
        (d / f"A0{s}.md").write_text("\n".join(L))


def report(a: argparse.Namespace) -> Path:
    cfg = load_v2_config(a.config)
    runs = Path(a.runs_dir)
    try:
        subjects = subject_list(cfg["test"]["subjects"])
    except (KeyError, TypeError, ValueError):
        subjects = list(range(1, 10))
    table, facts = collect(runs, a.seeds, a.students, subjects)
    costs = model_costs(cfg, a.students)
    results = Path(a.results).parent
    results.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(write_report(table, facts, cfg, a.seeds, a.students, subjects, costs, runs=runs))
    write_confusions(table, a.students, subjects, results)
    print(f"wrote {a.results}")
    return Path(a.results)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--students", nargs="+", default=["wide32", "control"])
    p.add_argument("--skip-train", action="store_true", help="only rebuild the report from existing runs")
    p.add_argument("--config", type=Path, default=CONFIG_V2)
    p.add_argument("--threads", type=int, default=int(os.environ.get("RB_THREADS", "1")))
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=None, help="passed to every run (smoke)")
    p.add_argument("--subset-cho", type=int, default=None, help="passed to every run (smoke)")
    p.add_argument("--runs-dir", type=Path, default=RUNS_V2)
    p.add_argument("--results", type=Path, default=RESULTS / "results_v2.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    a.runs_dir, a.config, a.results = a.runs_dir.resolve(), a.config.resolve(), a.results.resolve()
    if not a.skip_train:
        train_all(a)
    report(a)


if __name__ == "__main__":
    main()

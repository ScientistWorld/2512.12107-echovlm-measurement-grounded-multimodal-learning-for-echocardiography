"""Final hardening audit: regime search, reference set, variants, controls, shortcuts, official scoring.

Phases (resumable: finished trainings / score files are skipped):

  smoke   tiny trainings + every builder + every evaluator, written to runs_smoke/;
          aborts the job if the hardened evaluator rejects a genuine export.
  1       training-setup search for the paper's recipe. Candidate regimes differ
          only in data size (frames per sequence), epochs and batch size; each
          trains L_CLIP (baseline) and the full EchoVLM objective with 3 seeds.
          The regime is chosen by a rule fixed in advance: highest mean macro
          AUC of the full EchoVLM objective on the VISIBLE validation patients
          (the paper tunes on its validation split). The hidden test is scored
          for reporting only.
  2       on the selected regime: Table 6 ablations (3 seeds), 2 more seeds for
          the baseline and the reference, candidate improvements (augmentation,
          train+val data, 3-seed ensembles), weaker controls, shortcuts.
  3       official reference set (seed 0 of each Table 6 row) -> checkpoints/,
          scored by scripts/evaluate_{train,test}.sh and scripts/evaluate.sh.

Usage: python redteam/hardening/audit.py [--smoke-only] [--skip-smoke]
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
SCORES = HERE / "scores_v2"
DUMPS = HERE / "dumps"
LOGS = HERE / "logs_v2"
PY = sys.executable
M = [PY, "-m", "method.train"]
B = [PY, "-m", "baseline.clip_finetune"]
LEGACY_EVAL = ROOT / "eval" / "test" / "legacy_v1" / "evaluate_v1.py"

DATA_ALL = "data/downloads/camus_vlm"
DATA_F5 = "data/downloads/camus_vlm_f5"
REGIMES = {
    # name: (data dir, extra args). All keep the paper's lr 1e-4, wd 0.005, warmup 200, cosine,
    # lambda_view 0.5 / lambda_neg 0.1 and the paper's 512 batch unless stated.
    "R20all": (DATA_ALL, ["--epochs", "20", "--batch-size", "512"]),  # previous gym setting
    "R10all": (DATA_ALL, ["--epochs", "10", "--batch-size", "512"]),
    "R5all": (DATA_ALL, ["--epochs", "5", "--batch-size", "512"]),
    "R3all": (DATA_ALL, ["--epochs", "3", "--batch-size", "512"]),
    "R20f5": (DATA_F5, ["--epochs", "20", "--batch-size", "512"]),
    "R20f5b128": (DATA_F5, ["--epochs", "20", "--batch-size", "128"]),
}
METHODS = {
    "full": (M, ["--lambda-view", "0.5", "--lambda-neg", "0.1"]),
    "clip": (B, []),
    "view": (M, ["--lambda-view", "0.5", "--lambda-neg", "0"]),
    "neg": (M, ["--lambda-view", "0", "--lambda-neg", "0.1"]),
    "aug": (M, ["--lambda-view", "0.5", "--lambda-neg", "0.1", "--augment"]),
    "trainval": (M, ["--lambda-view", "0.5", "--lambda-neg", "0.1", "--train-splits", "train", "val"]),
}
# Exports from the previous audit (same code path, same data) reused instead of retrained.
REUSE = {
    "full__R20all__s0": "echovlm_full_s0",
    "full__R20all__s1": "echovlm_full_s1",
    "full__R20all__s2": "echovlm_full_s2",
    "clip__R20all__s0": "clip_only_s0",
    "clip__R20all__s1": "clip_only_s1",
    "clip__R20all__s2": "clip_only_s2",
    "full__R5all__s0": "echovlm_full_ep5_s0",
}
OFFICIAL = {"echovlm_full": "full", "clip_only": "clip", "echovlm_view": "view", "echovlm_negation": "neg"}


def log(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def run(cmd: list[str], logfile: Path) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def gpu_ids() -> list[int]:
    try:
        import torch

        return list(range(min(2, torch.cuda.device_count())))
    except Exception:
        return []


def run_name(method: str, regime: str, seed: int) -> str:
    return f"{method}__{regime}__s{seed}"


def train_cmd(method: str, regime: str, seed: int, smoke: bool = False) -> list[str]:
    base, margs = METHODS[method]
    data, rargs = REGIMES[regime] if not smoke else (DATA_ALL, [])
    cmd = base + margs + ["--data-dir", data, "--seed", str(seed)]
    if method == "clip":  # baseline has its own argument parser
        cmd = base + ["--data-dir", data, "--seed", str(seed)]
    if smoke:
        return cmd + ["--epochs", "1", "--batch-size", "64", "--warmup-steps", "2", "--limit-images", "256"]
    return cmd + rargs


def score(runs: Path, name: str, device: str, scores: Path, dumps: Path, logs: Path, legacy: bool = False) -> None:
    sides = [("test", "eval/test/evaluate.py", ["--dump-dir", str(dumps / "test")])]
    sides.append(("train", "eval/train/evaluate.py", ["--dump-dir", str(dumps / "train")]))
    if legacy and LEGACY_EVAL.is_file():
        sides.append(("legacy_test", str(LEGACY_EVAL), []))
    for side, script, extra in sides:
        out = scores / f"{name}.{side}.json"
        if out.is_file():
            continue
        rc = run([PY, script, "--root", str(runs), "--only", name, "--device", device, "--output", str(out)] + extra,
                 logs / f"{name}.eval.log")
        log({"scored": name, "side": side, "rc": rc})


def pool_run(jobs: list[tuple[str, list[str] | None, bool]], runs: Path, scores: Path, dumps: Path, logs: Path) -> None:
    """jobs: (name, train command or None, legacy-score?). Two workers per GPU."""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    devices = [f"cuda:{g}" for g in gpu_ids() for _ in range(2)] or ["cpu"]

    def worker(device: str) -> None:
        while True:
            try:
                name, cmd, legacy = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            if cmd is not None and not (runs / name / "export" / "text_encoder.pt").is_file():
                rc = run(cmd + ["--output-dir", str(runs / name), "--device", device], logs / f"{name}.train.log")
                log({"trained": name, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)})
            if (runs / name).is_dir():
                score(runs, name, device, scores, dumps, logs, legacy)

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def status(scores: Path, name: str, side: str) -> dict:
    f = scores / f"{name}.{side}.json"
    if not f.is_file():
        return {}
    return json.loads(f.read_text())["_meta"]["method_status"].get(name, {})


# ----------------------------------------------------------------------------- phases


def smoke() -> None:
    runs, scores, dumps, logs = HERE / "runs_smoke", HERE / "scores_smoke", HERE / "dumps_smoke", HERE / "logs_smoke"
    for d in (runs, scores, dumps, logs):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    jobs = [
        ("smoke_full", train_cmd("full", "", 0, smoke=True), False),
        ("smoke_clip", train_cmd("clip", "", 0, smoke=True), False),
        ("smoke_aug", train_cmd("aug", "", 0, smoke=True), False),
    ]
    pool_run(jobs, runs, scores, dumps, logs)
    rc = run([PY, str(HERE / "make_variants.py"), "--members", str(runs / "smoke_full"), str(runs / "smoke_aug"),
              "--out", str(runs / "smoke_ens")], logs / "variants.log")
    log({"smoke_ensemble_built": rc})
    rc = run([PY, str(HERE / "make_shortcuts.py"), "--ref", str(runs / "smoke_full"), "--out", str(runs)], logs / "shortcuts.log")
    log({"smoke_shortcuts_built": rc})
    names = ["smoke_ens"] + sorted(p.name for p in runs.glob("sc_*"))
    pool_run([(n, None, n in ("sc_file_reader", "sc_stateful")) for n in names], runs, scores, dumps, logs)
    report = {}
    for n in ["smoke_full", "smoke_clip", "smoke_aug"] + names:
        report[n] = {side: {k: status(scores, n, side).get(k) for k in ("status", "reason", "macro_auc")}
                     for side in ("test", "train", "legacy_test")}
    (HERE / "smoke_report.json").write_text(json.dumps(report, indent=2))
    log({"smoke_report": report})
    bad = []
    for n in ("smoke_full", "smoke_clip", "smoke_aug", "smoke_ens"):
        for side in ("test", "train"):
            st = status(scores, n, side)
            if not st or st.get("status") == "failed":
                bad.append(f"{n}/{side}: {st.get('reason')}")
    if bad:
        log({"SMOKE_FAILED": bad})
        sys.exit(3)
    log({"smoke": "passed"})


def reuse_previous() -> None:
    for new, old in list(REUSE.items()) + [("untrained", "untrained")]:
        src, dst = RUNS / old, RUNS / new
        if dst.exists() or not (src / "export" / "text_encoder.pt").is_file():
            continue
        src.rename(dst)
        log({"reused": new, "from": old})


def phase1() -> str:
    jobs = []
    for regime in REGIMES:
        for method in ("full", "clip"):
            for seed in (0, 1, 2):
                name = run_name(method, regime, seed)
                jobs.append((name, train_cmd(method, regime, seed), False))
    pool_run(jobs, RUNS, SCORES, DUMPS, LOGS)
    table = {}
    for regime in REGIMES:
        val = [status(SCORES, run_name("full", regime, s), "train").get("macro_auc") for s in (0, 1, 2)]
        val = [v for v in val if v is not None]
        table[regime] = {"full_val_macro_auc_mean": statistics.mean(val) if val else -1.0, "n": len(val)}
    selected = max(table, key=lambda r: table[r]["full_val_macro_auc_mean"])
    (HERE / "selected_regime.json").write_text(json.dumps({"selected": selected, "rule": "max mean visible-val macro AUC of the full objective over seeds 0-2", "table": table}, indent=2))
    log({"selected_regime": selected, "table": table})
    return selected


def phase2(g: str) -> None:
    jobs = []
    for method in ("view", "neg", "aug", "trainval"):
        for seed in (0, 1, 2):
            jobs.append((run_name(method, g, seed), train_cmd(method, g, seed), False))
    for method in ("full", "clip"):
        for seed in (3, 4):
            jobs.append((run_name(method, g, seed), train_cmd(method, g, seed), False))
    jobs.append(("untrained", M + ["--epochs", "0", "--seed", "0"], False))
    jobs.append((run_name("full", g, 0), train_cmd("full", g, 0), True))  # legacy score of the reference
    pool_run(jobs, RUNS, SCORES, DUMPS, LOGS)
    for kind in ("full", "clip"):
        name = f"ens3_{kind}__{g}"
        if not (RUNS / name / "export" / "text_encoder.pt").is_file():
            members = [str(RUNS / run_name(kind, g, s)) for s in (0, 1, 2)]
            rc = run([PY, str(HERE / "make_variants.py"), "--members", *members, "--out", str(RUNS / name)], LOGS / "variants.log")
            log({"ensemble_built": name, "rc": rc})
    ref = RUNS / run_name("full", g, 0)
    if not (RUNS / "sc_handcrafted" / "export" / "text_encoder.pt").is_file():
        rc = run([PY, str(HERE / "make_shortcuts.py"), "--ref", str(ref), "--out", str(RUNS),
                  "--train-data", REGIMES[g][0]], LOGS / "shortcuts.log")
        log({"shortcuts_built": rc})
    names = [f"ens3_full__{g}", f"ens3_clip__{g}"] + sorted(p.name for p in RUNS.glob("sc_*"))
    pool_run([(n, None, n.startswith("sc_")) for n in names], RUNS, SCORES, DUMPS, LOGS)


def phase3(g: str) -> None:
    ckpt = ROOT / "checkpoints"
    for official, method in OFFICIAL.items():
        src, dst = RUNS / run_name(method, g, 0), ckpt / official
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        shutil.copytree(src / "export", dst / "export")
        for f in ("run_config.json", "history.json"):
            if (src / f).is_file():
                shutil.copyfile(src / f, dst / f)
    for script in ("scripts/evaluate_train.sh", "scripts/evaluate_test.sh", "scripts/evaluate.sh"):
        rc = run(["bash", script], LOGS / "official.log")
        log({"official": script, "rc": rc})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    for d in (RUNS, SCORES, DUMPS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    if not args.skip_smoke and not (HERE / "smoke_report.json").is_file():
        smoke()
    if args.smoke_only:
        return
    reuse_previous()
    g = phase1()
    phase2(g)
    phase3(g)


if __name__ == "__main__":
    main()

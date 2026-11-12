"""Train and score every hardening run (reference seeds, ablations, variants, shortcuts).

Runs are written to redteam/hardening/runs/<name>/ and scored with the
hidden-test evaluator (eval/test/evaluate.py) and the visible evaluator
(eval/train/evaluate.py), one run at a time, into
redteam/hardening/scores/<name>.{test,train}.json. Finished runs are skipped,
so the job can be resumed. Training jobs are spread over the visible GPUs.

Usage: python redteam/hardening/orchestrate.py [--smoke] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SCORES = HERE / "scores"
LOGS = HERE / "logs"

M = [sys.executable, "-m", "method.train"]
B = [sys.executable, "-m", "baseline.clip_finetune"]


def plan(smoke: bool) -> list[tuple[str, list[str]]]:
    if smoke:
        quick = ["--epochs", "1", "--batch-size", "128", "--warmup-steps", "4", "--limit-images", "1024"]
        return [
            ("smoke_full", M + quick),
            ("smoke_clip", B + quick),
            ("smoke_untrained", M + ["--epochs", "0"]),
        ]
    runs = []
    # Reference reproduction (paper Table 6 rows, paper settings: 20 epochs, batch 512).
    runs.append(("echovlm_full_s0", M + ["--seed", "0"]))
    runs.append(("clip_only_s0", B + ["--seed", "0"]))
    runs.append(("echovlm_view_s0", M + ["--lambda-view", "0.5", "--lambda-neg", "0", "--seed", "0"]))
    runs.append(("echovlm_negation_s0", M + ["--lambda-view", "0", "--lambda-neg", "0.1", "--seed", "0"]))
    # Seed replicates of the reference and the baseline.
    for s in (1, 2):
        runs.append((f"echovlm_full_s{s}", M + ["--seed", str(s)]))
        runs.append((f"clip_only_s{s}", B + ["--seed", str(s)]))
    # Weaker controls.
    runs.append(("untrained", M + ["--epochs", "0"]))
    runs.append(("echovlm_full_ep5_s0", M + ["--epochs", "5", "--seed", "0"]))
    # Candidate improvements (same objective, better optimisation / more data).
    runs.append(("echovlm_full_ep50_s0", M + ["--epochs", "50", "--seed", "0"]))
    runs.append(("clip_only_ep50_s0", B + ["--epochs", "50", "--seed", "0"]))
    runs.append(("echovlm_full_bs128_s0", M + ["--batch-size", "128", "--seed", "0"]))
    runs.append(("echovlm_full_trainval_s0", M + ["--train-splits", "train", "val", "--seed", "0"]))
    for s in (1, 2):
        runs.append((f"echovlm_full_ep50_s{s}", M + ["--epochs", "50", "--seed", str(s)]))
        runs.append((f"clip_only_ep50_s{s}", B + ["--epochs", "50", "--seed", str(s)]))
    return runs


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def score(name: str, device: str) -> None:
    for side, script in (("test", "eval/test/evaluate.py"), ("train", "eval/train/evaluate.py")):
        out = SCORES / f"{name}.{side}.json"
        if out.is_file():
            continue
        rc = run(
            [sys.executable, script, "--root", str(RUNS), "--only", name, "--device", device, "--output", str(out)],
            LOGS / f"{name}.eval.log",
        )
        print(json.dumps({"scored": name, "side": side, "rc": rc}), flush=True)


def worker(gpu: int, jobs: "queue.Queue[tuple[str, list[str]]]") -> None:
    device = f"cuda:{gpu}"
    while True:
        try:
            name, cmd = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        if not (RUNS / name / "export" / "text_encoder.pt").is_file():
            rc = run(cmd + ["--output-dir", str(RUNS / name), "--device", device], LOGS / f"{name}.train.log")
            print(json.dumps({"trained": name, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
        score(name, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for d in (RUNS, SCORES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    runs = plan(args.smoke)
    if args.only:
        runs = [r for r in runs if r[0] in set(args.only)]
    n_gpu = max(1, gpu_count())
    jobs: queue.Queue = queue.Queue()
    for r in runs:
        jobs.put(r)
    threads = [threading.Thread(target=worker, args=(g, jobs)) for g in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Shortcuts are derived from the seed-0 reference export.
    ref = RUNS / ("smoke_full" if args.smoke else "echovlm_full_s0")
    if (ref / "export" / "text_encoder.pt").is_file():
        rc = run([sys.executable, str(HERE / "make_shortcuts.py"), "--ref", str(ref), "--out", str(RUNS)], LOGS / "shortcuts.log")
        print(json.dumps({"shortcuts_built": rc}), flush=True)
        for d in sorted(RUNS.glob("sc_*")):
            score(d.name, "cuda:0" if gpu_count() else "cpu")


if __name__ == "__main__":
    main()

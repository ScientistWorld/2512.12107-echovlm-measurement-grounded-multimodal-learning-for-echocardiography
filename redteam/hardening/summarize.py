"""Summarise hardening runs: metrics, seed spread, paired gaps and rank deltas.

Rank delta of a run vs the reference (echovlm_full_s0) follows the stated rule:
    sum_exp weight_exp * sum_metric |coef| * (improvement in the better direction) / sd_metric
where sd_metric is the population SD of that metric across the cohort of runs
(reported twice: over genuine non-failed runs, and over all runs incl. shortcuts).

Usage: python redteam/hardening/summarize.py [--reference echovlm_full_s0]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_JSON = HERE.parents[1] / "scoring" / "reference.json"


def load(side: str) -> tuple[dict, dict]:
    runs, status = {}, {}
    for f in sorted((HERE / "scores").glob(f"*.{side}.json")):
        data = json.loads(f.read_text())
        for exp, block in data["experiments"].items():
            for name, metrics in block["results"].items():
                runs.setdefault(name, {})[exp] = metrics
        status.update(data.get("_meta", {}).get("method_status", {}))
    return runs, status


def scored_metrics(ref: dict) -> list[tuple[str, str, float, float, bool]]:
    out = []
    for exp, e in ref["experiments"].items():
        for m, d in e["metrics"].items():
            if d.get("coefficient", 0):
                out.append((exp, m, float(e["weight"]), abs(float(d["coefficient"])), bool(d["higher_is_better"])))
    return out


def rank_deltas(runs: dict, reference: str, metrics, cohort: list[str]) -> dict:
    sds = {}
    for exp, m, *_ in metrics:
        vals = [runs[r][exp][m] for r in cohort if exp in runs[r] and m in runs[r][exp]]
        sds[(exp, m)] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    out = {}
    for name, r in runs.items():
        total = 0.0
        for exp, m, w, c, hib in metrics:
            sd = sds[(exp, m)]
            if sd <= 0 or exp not in r or m not in r[exp]:
                continue
            diff = r[exp][m] - runs[reference][exp][m]
            total += w * c * (diff if hib else -diff) / sd
        out[name] = round(total, 4)
    return {"deltas": out, "sd": {f"{e}.{m}": round(v, 4) for (e, m), v in sds.items()}}


def seed_stats(runs: dict, names: list[str], metrics) -> dict:
    out = {}
    for exp, m, *_ in metrics:
        vals = [runs[n][exp][m] for n in names if n in runs]
        if vals:
            out[f"{exp}.{m}"] = {
                "mean": round(statistics.mean(vals), 3),
                "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "values": [round(v, 3) for v in vals],
            }
    return out


def paired(runs: dict, a: str, b: str, seeds, metrics) -> dict:
    out = {}
    for exp, m, *_ in metrics:
        diffs = [runs[f"{a}_s{s}"][exp][m] - runs[f"{b}_s{s}"][exp][m] for s in seeds if f"{a}_s{s}" in runs and f"{b}_s{s}" in runs]
        if diffs:
            out[f"{exp}.{m}"] = {
                "mean_diff": round(statistics.mean(diffs), 3),
                "diffs": [round(d, 3) for d in diffs],
                "all_same_sign": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="echovlm_full_s0")
    args = ap.parse_args()
    ref = json.loads(REF_JSON.read_text())
    metrics = scored_metrics(ref)
    summary = {"reference": args.reference, "scored_metrics": [f"{e}.{m}" for e, m, *_ in metrics]}
    for side in ("test", "train"):
        runs, status = load(side)
        if args.reference not in runs:
            continue
        genuine = [n for n in runs if not n.startswith("sc_") and status.get(n, {}).get("status") == "ok"]
        summary[side] = {
            "status": {n: status.get(n, {}).get("status") for n in runs},
            "gate_p": {n: status.get(n, {}).get("permutation_p_value") for n in runs},
            "metrics": runs,
            "rank_vs_reference_genuine_sd": rank_deltas(runs, args.reference, metrics, genuine),
            "rank_vs_reference_all_sd": rank_deltas(runs, args.reference, metrics, list(runs)),
            "seeds": {
                "echovlm_full_ep20": seed_stats(runs, [f"echovlm_full_s{s}" for s in range(3)], metrics),
                "clip_only_ep20": seed_stats(runs, [f"clip_only_s{s}" for s in range(3)], metrics),
                "echovlm_full_ep50": seed_stats(runs, [f"echovlm_full_ep50_s{s}" for s in range(3)], metrics),
                "clip_only_ep50": seed_stats(runs, [f"clip_only_ep50_s{s}" for s in range(3)], metrics),
            },
            "paired_full_minus_clip_ep20": paired(runs, "echovlm_full", "clip_only", range(3), metrics),
            "paired_full_minus_clip_ep50": paired(runs, "echovlm_full_ep50", "clip_only_ep50", range(3), metrics),
            "paired_ep50_minus_ep20_full": {
                k: v
                for k, v in paired(
                    {**{f"a_s{s}": runs.get(f"echovlm_full_ep50_s{s}") for s in range(3)}, **{f"b_s{s}": runs.get(f"echovlm_full_s{s}") for s in range(3)}},
                    "a",
                    "b",
                    [s for s in range(3) if runs.get(f"echovlm_full_ep50_s{s}") and runs.get(f"echovlm_full_s{s}")],
                    metrics,
                ).items()
            },
        }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = ["| run | status | " + " | ".join(summary["scored_metrics"]) + " | rank Δ (genuine SD) | rank Δ (all SD) |", "|---" * (len(metrics) + 4) + "|"]
    test = summary.get("test", {})
    for name in sorted(test.get("metrics", {})):
        r = test["metrics"][name]
        vals = [f"{r[e][m]:.2f}" for e, m, *_ in metrics]
        lines.append(
            f"| {name} | {test['status'][name]} | " + " | ".join(vals)
            + f" | {test['rank_vs_reference_genuine_sd']['deltas'][name]:+.3f} | {test['rank_vs_reference_all_sd']['deltas'][name]:+.3f} |"
        )
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

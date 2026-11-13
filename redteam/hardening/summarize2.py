"""Summarise the final hardening audit (scores_v2/, dumps/).

* regime search: per regime, full vs CLIP-only, 3 seeds, visible-val and hidden-test macro AUC
* selected regime: every run (ablations, seeds, variants, controls) with its scored metrics
* paired patient-level bootstrap of the hidden-test macro AUC difference vs the reference
  (every run sees the same 1,500 hidden images; patients are resampled jointly)
* rank delta vs the reference under the stated rule:
      sum_exp weight * sum_metric |coef| * (improvement in the better direction) / sd(metric over cohort)
  cohort = all genuine (non-shortcut) runs scored on the hidden test
* shortcuts: hardened evaluator vs the previous (legacy) evaluator

Usage: python redteam/hardening/summarize2.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORES = HERE / "scores_v2"
DUMPS = HERE / "dumps" / "test"
REF_JSON = ROOT / "scoring" / "reference.json"
NAME_RE = re.compile(r"^(?P<method>[a-z0-9]+)__(?P<regime>[A-Za-z0-9]+)__s(?P<seed>\d+)$")
DISEASES = ("lv_systolic_dysfunction", "lv_dilation", "la_enlargement")
N_BOOT = 2000


def load(side: str) -> tuple[dict, dict]:
    runs, status = {}, {}
    for f in sorted(SCORES.glob(f"*.{side}.json")):
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


def rank_deltas(runs: dict, reference: str, metrics, cohort: list[str]) -> tuple[dict, dict]:
    sds = {}
    for exp, m, *_ in metrics:
        vals = [runs[r][exp][m] for r in cohort if r in runs and exp in runs[r] and m in runs[r][exp]]
        sds[(exp, m)] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    out = {}
    for name, r in runs.items():
        total = 0.0
        for exp, m, w, c, hib in metrics:
            sd = sds[(exp, m)]
            if sd <= 0 or exp not in r or m not in r[exp] or reference not in runs:
                continue
            diff = r[exp][m] - runs[reference][exp][m]
            total += w * c * (diff if hib else -diff) / sd
        out[name] = round(total, 4)
    return out, {f"{e}.{m}": round(v, 4) for (e, m), v in sds.items()}


def ms(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


# ----------------------------------------------------------------------------- paired bootstrap


def load_labels() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "eval" / "test" / "hidden_test.npz", allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    recs = meta["records"]
    y = np.array([[r["labels"][d] for d in DISEASES] for r in recs], dtype=np.int64)
    patients = sorted({r["patient"] for r in recs})
    index = {p: i for i, p in enumerate(patients)}
    pid = np.array([index[r["patient"]] for r in recs])
    return y, pid


def macro_auc(scores: np.ndarray, y: np.ndarray) -> float:
    aucs = []
    for j in range(y.shape[1]):
        s, t = scores[:, j], y[:, j]
        n1 = int(t.sum())
        n0 = len(t) - n1
        if n1 == 0 or n0 == 0:
            aucs.append(0.5)
            continue
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        bounds = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
        avg = (bounds[:-1] + 1 + bounds[1:]) / 2.0
        ranks = np.empty(len(s))
        ranks[order] = np.repeat(avg, np.diff(bounds))
        aucs.append((ranks[t == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return 100.0 * float(np.mean(aucs))


def paired_bootstrap(names: list[str], reference: str) -> dict:
    if not (DUMPS / f"{reference}.npz").is_file():
        return {}
    y, pid = load_labels()
    n_pat = pid.max() + 1
    members = [np.flatnonzero(pid == p) for p in range(n_pat)]
    rng = np.random.default_rng(12345)
    samples = [np.concatenate([members[p] for p in rng.integers(0, n_pat, n_pat)]) for _ in range(N_BOOT)]
    ref_s = np.load(DUMPS / f"{reference}.npz")["disease_scores"].astype(np.float64)
    ref_boot = np.array([macro_auc(ref_s[ix], y[ix]) for ix in samples])
    out = {}
    for name in names:
        f = DUMPS / f"{name}.npz"
        if not f.is_file() or name == reference:
            continue
        s = np.load(f)["disease_scores"].astype(np.float64)
        d = np.array([macro_auc(s[ix], y[ix]) for ix in samples]) - ref_boot
        out[name] = {
            "delta_macro_auc": round(macro_auc(s, y) - macro_auc(ref_s, y), 3),
            "ci95": [round(float(np.quantile(d, 0.025)), 3), round(float(np.quantile(d, 0.975)), 3)],
            "se": round(float(d.std()), 3),
            "p_not_better": round(float(np.mean(d <= 0)), 4),
        }
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    ref_def = json.loads(REF_JSON.read_text())
    metrics = scored_metrics(ref_def)
    test, st_test = load("test")
    val, st_val = load("train")
    legacy, st_legacy = load("legacy_test")
    sel = json.loads((HERE / "selected_regime.json").read_text())
    g = sel["selected"]
    reference = f"full__{g}__s0"

    parsed = {n: NAME_RE.match(n).groupdict() for n in test if NAME_RE.match(n)}
    genuine = [n for n in test if not n.startswith("sc_")]
    ranks, sds = rank_deltas(test, reference, metrics, genuine)
    boot = paired_bootstrap(sorted(test), reference)

    def raw_auc(st: dict, n: str):
        v = st.get(n, {}).get("macro_auc")
        return None if v is None else 100.0 * v

    lines = [f"# Final hardening audit (selected regime: `{g}`; reference: `{reference}`)", ""]
    lines += ["## Regime search (macro AUC, raw before the gate; mean ± sd over seeds 0-2)", "",
              "| regime | full val | clip val | full test | clip test | paired full−clip test (per seed) | full gate pass (test) |",
              "|---|---|---|---|---|---|---|"]
    regimes = sorted({p["regime"] for p in parsed.values()})
    regime_table = {}
    for r in regimes:
        row = {}
        for m in ("full", "clip"):
            names = [f"{m}__{r}__s{s}" for s in (0, 1, 2)]
            row[m] = {
                "val": [raw_auc(st_val, n) for n in names],
                "test": [raw_auc(st_test, n) for n in names],
                "pass": [st_test.get(n, {}).get("status") == "ok" for n in names],
            }
        diffs = [a - b for a, b in zip(row["full"]["test"], row["clip"]["test"]) if a is not None and b is not None]
        regime_table[r] = {**row, "paired_diff": diffs}
        lines.append(f"| {r} | {ms(row['full']['val'])} | {ms(row['clip']['val'])} | {ms(row['full']['test'])} | "
                     f"{ms(row['clip']['test'])} | {ms(diffs)} | {sum(row['full']['pass'])}/3 |")

    lines += ["", f"## Selected regime `{g}`: hidden-test metrics (scored values; gate failures are 0)", "",
              "| run | status | AUC | prec | recall | R@5 | R@10 | kNN F1 | val AUC (raw) | ΔAUC vs ref [95% CI] | rank Δ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    sel_names = sorted(n for n in test if (n in parsed and parsed[n]["regime"] == g) or n.startswith("ens3_") or n == "untrained")
    for n in sel_names:
        t = test[n]
        b = boot.get(n, {})
        ci = f"{b['delta_macro_auc']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]" if b else "ref" if n == reference else "—"
        va = raw_auc(st_val, n)
        lines.append(
            f"| {n} | {st_test.get(n, {}).get('status')} | {t['disease_zero_shot']['auc']:.2f} | {t['disease_zero_shot']['precision']:.2f} | "
            f"{t['disease_zero_shot']['recall']:.2f} | {t['image_text_retrieval']['recall_at_5']:.2f} | {t['image_text_retrieval']['recall_at_10']:.2f} | "
            f"{t['view_knn']['f1']:.2f} | {'—' if va is None else f'{va:.2f}'} | {ci} | {ranks.get(n, 0):+.3f} |")

    # Per-method seed statistics in the selected regime.
    seedstats = {}
    for m in ("full", "clip", "view", "neg", "aug", "trainval"):
        names = sorted(n for n, p in parsed.items() if p["regime"] == g and p["method"] == m)
        if not names:
            continue
        seedstats[m] = {
            "n": len(names),
            "test_auc": ms([raw_auc(st_test, n) for n in names]),
            "val_auc": ms([raw_auc(st_val, n) for n in names]),
            "rank": ms([ranks.get(n) for n in names]),
            "r10": ms([test[n]["image_text_retrieval"]["recall_at_10"] for n in names]),
            "knn": ms([test[n]["view_knn"]["f1"] for n in names]),
            "gate_pass": sum(st_test.get(n, {}).get("status") == "ok" for n in names),
        }
    lines += ["", "## Selected regime: seed statistics (raw macro AUC; rank Δ vs reference)", "",
              "| method | n | test AUC | val AUC | R@10 | kNN F1 | rank Δ | gate pass |", "|---|---|---|---|---|---|---|---|"]
    for m, s in seedstats.items():
        lines.append(f"| {m} | {s['n']} | {s['test_auc']} | {s['val_auc']} | {s['r10']} | {s['knn']} | {s['rank']} | {s['gate_pass']}/{s['n']} |")

    lines += ["", "## Shortcuts: hardened evaluator vs previous evaluator", "",
              "| shortcut | hardened status | hardened AUC / R@10 / kNN | rank Δ | previous status | previous AUC / R@10 / kNN |",
              "|---|---|---|---|---|---|"]
    for n in sorted(x for x in test if x.startswith("sc_")):
        t, l = test[n], legacy.get(n)
        hs = st_test.get(n, {})
        ls = st_legacy.get(n, {})
        lt = (f"{l['disease_zero_shot']['auc']:.2f} / {l['image_text_retrieval']['recall_at_10']:.2f} / {l['view_knn']['f1']:.2f}" if l else "—")
        lines.append(f"| {n} | {hs.get('status')}: {str(hs.get('reason', ''))[:70]} | {t['disease_zero_shot']['auc']:.2f} / "
                     f"{t['image_text_retrieval']['recall_at_10']:.2f} / {t['view_knn']['f1']:.2f} | {ranks.get(n, 0):+.3f} | "
                     f"{ls.get('status', '—')} | {lt} |")
    if reference in legacy:
        l = legacy[reference]
        lines.append(f"| (reference under previous evaluator) | {st_legacy.get(reference, {}).get('status')} | — | — | — | "
                     f"{l['disease_zero_shot']['auc']:.2f} / {l['image_text_retrieval']['recall_at_10']:.2f} / {l['view_knn']['f1']:.2f} |")
    lines += ["", f"Metric SDs over the genuine cohort ({len(genuine)} runs): `{json.dumps(sds)}`"]

    summary = {
        "selected": sel,
        "reference": reference,
        "regime_table": regime_table,
        "seed_stats": seedstats,
        "bootstrap_vs_reference": boot,
        "rank_deltas": ranks,
        "metric_sd": sds,
        "status_test": st_test,
        "status_val": st_val,
        "status_legacy": st_legacy,
    }
    (HERE / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    (HERE / "summary_v2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

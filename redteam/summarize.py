"""Rank the measured audit cohort using direction, absolute weight and stddev."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "redteam"
EXPERIMENT = "toy_disease_zero_shot"


def main():
    definition = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"][EXPERIMENT]
    weights = {k: v for k, v in definition["metrics"].items() if abs(v["coefficient"]) > 0}
    rows = {}
    status = {}
    for folder in sorted(AUDIT.iterdir()):
        meta = folder / "run_metadata.json"
        if not meta.exists():
            continue
        status[folder.name] = json.loads(meta.read_text())["test_exit_code"]
        scores = folder / "scores_test.json"
        if status[folder.name] or not scores.exists():
            continue
        results = json.loads(scores.read_text())["experiments"][EXPERIMENT]["results"]
        if folder.name == "reference_cached":
            rows["reference"] = results["echovlm_full"]
            rows["original_clip_only"] = results["clip_only"]
        rows[folder.name] = results.get(folder.name, {})
    # Count reference only once. The copy under reference_cached is a rerun
    # consistency check, not an independent method in the normalization cohort.
    rows.pop("reference_cached", None)
    standard_deviations = {
        key: statistics.pstdev([values[key] for values in rows.values() if key in values])
        for key in weights
    }
    ref = rows["reference"]

    def rank(values):
        contributions = {}
        for key, definition_metric in weights.items():
            if key not in values:
                continue
            std = standard_deviations[key]
            direction = 1 if definition_metric["higher_is_better"] else -1
            contributions[key] = (abs(definition_metric["coefficient"]) * direction *
                                  (values[key] - ref[key]) / std if std else 0)
        return {"score": definition["weight"] * sum(contributions.values()),
                "contributions": contributions, "metrics": values}

    ranked = {name: rank(values) for name, values in rows.items()}
    seed_names = [f"reference_seed{seed}" for seed in (7, 17, 27)]
    seed_scores = [ranked[name]["score"] for name in seed_names if name in ranked]
    spread = {key: {"min": min(rows[name][key] for name in seed_names if name in rows),
                    "max": max(rows[name][key] for name in seed_names if name in rows)}
              for key in weights}
    for value in spread.values():
        value["range"] = value["max"] - value["min"]
    result = {"reference": ref, "weights": weights, "normalization": {
        "cohort": list(rows), "population_standard_deviation": standard_deviations,
        "note": "Actual future ranks depend on their cohort; metric-wise dominance does not."},
        "ranked": ranked, "run_exit_codes": status, "reference_seed_spread": spread,
        "reference_rank_spread": {"min": min(seed_scores), "max": max(seed_scores),
                                  "range": max(seed_scores) - min(seed_scores)}}
    (AUDIT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, value in sorted(ranked.items(), key=lambda item: item[1]["score"], reverse=True):
        print(name, round(value["score"], 6), {key: round(value["metrics"][key], 5)
              for key in weights if key in value["metrics"]})
    print("SEED SPREAD", json.dumps(spread))


if __name__ == "__main__":
    main()

"""Lightweight consistency checks of saved audit evidence (no model execution)."""
import json
from pathlib import Path

from redteam.run_case import CASES

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "redteam"
EXPERIMENT = "toy_disease_zero_shot"
SCORED = {"auc", "precision", "recall", "negation_cosine"}


def main():
    successful, failed, omitted = [], [], []
    for name in CASES:
        folder = AUDIT / name
        assert (folder / "run.sh").is_file(), name
        meta = json.loads((folder / "run_metadata.json").read_text())
        assert meta["test_command"] == "bash scripts/evaluate_test.sh", name
        assert meta["evaluator_sha256_before"] == meta["evaluator_sha256_after"], name
        path = folder / "scores_test.json"
        if meta["test_exit_code"]:
            assert not path.exists(), name
            assert (folder / "failure.json").is_file(), name
            failed.append(name)
            continue
        exp = json.loads(path.read_text())["experiments"][EXPERIMENT]
        assert exp["slice"] == "test", name
        assert set(exp["results"]) == set(exp["split"]["input_paths"]), name
        for values in exp["results"].values():
            assert SCORED <= values.keys(), name
        if name == "shortcut_missing_artifact":
            assert name not in exp["results"], name
            omitted.append(name)
        else:
            assert name in exp["results"], name
        successful.append(name)
    summary = json.loads((AUDIT / "summary.json").read_text())
    ranked = summary["ranked"]
    assert ranked["shortcut_constant_scaled_failure"]["score"] > 0
    assert ranked["shortcut_constant_scaled_failure"]["metrics"]["auc"] == 50
    assert ranked["shortcut_forged_scores"]["metrics"]["auc"] == 50
    assert summary["reference_seed_spread"]["auc"]["range"] == 75
    report = {"case_count": len(CASES), "successful_hidden_tests": successful,
              "failed_hidden_tests": failed, "omitted_candidates": omitted,
              "all_evaluator_hashes_unchanged": True,
              "all_successful_method_results_have_all_four_scored_metrics": True,
              "all_present_discovered_input_paths_are_in_results": True}
    (AUDIT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

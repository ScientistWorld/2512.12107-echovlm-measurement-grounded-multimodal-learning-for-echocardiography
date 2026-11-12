"""Real artifact-based audit runs. Never changes evaluator code or environment."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "redteam"
CASES = {
    "reference_cached": {},
    "reference_seed7": {"epochs": 20, "seed": 7},
    "reference_seed17": {"epochs": 20, "seed": 17},
    "reference_seed27": {"epochs": 20, "seed": 27},
    "stronger_training100": {"epochs": 100, "seed": 7},
    "stronger_training200": {"epochs": 200, "seed": 7},
    "weaker_untrained": {"epochs": 0, "seed": 7},
    "weaker_clip_only": {"epochs": 20, "seed": 7, "lambda_view": 0, "lambda_neg": 0},
    "shortcut_visible_answers": {},
    "shortcut_generator_answers": {},
    "shortcut_text_only": {},
    "shortcut_bruteforce_answers": {},
    "shortcut_forged_labels": {},
    "shortcut_norm_scaling": {},
    "shortcut_constant_antipodes": {},
    "shortcut_constant_scaled_failure": {},
    "shortcut_empty_vectors": {},
    "shortcut_forged_scores": {},
    "shortcut_baseline_degradation": {},
    "shortcut_row_selection": {},
    "shortcut_missing_artifact": {},
    "shortcut_empty_artifact": {},
    "shortcut_null_vectors": {},
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def probe(row):
    keys = sorted(row["disease_labels"])
    key = keys[row["source_index"] % len(keys)]
    return key, int(row["disease_labels"][key])


def base_rows():
    result = []
    for row in read_rows(ROOT / "data/echo_toy/test.jsonl"):
        disease, label = probe(row)
        result.append({"id": row["id"], "disease": disease, "label": label,
                       "caption": row["caption"], "image_embedding": [1.0],
                       "text_embedding": [1.0], "neg_text_embedding": [-1.0]})
    return result


def margin(row):
    return sum(a * b for a, b in zip(row["image_embedding"], row["text_embedding"])) - sum(
        a * b for a, b in zip(row["image_embedding"], row["neg_text_embedding"]))


def run_logged(cmd, log):
    with Path(log).open("w") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


@contextlib.contextmanager
def isolated_artifacts():
    """Isolate a run on disk, including stale scores, then restore original bytes."""
    state = AUDIT / ".state"
    state.mkdir(exist_ok=True)
    saved = state / "original_checkpoints"
    if saved.exists():
        raise RuntimeError("An interrupted audit left .state/original_checkpoints; restore it first.")
    checkpoints = ROOT / "checkpoints"
    had_checkpoints = checkpoints.exists()
    score_paths = [ROOT / "scoring/scores_test.json", ROOT / "scoring/scores_train.json"]
    saved_scores = {p: p.read_bytes() if p.exists() else None for p in score_paths}
    if had_checkpoints:
        checkpoints.rename(saved)
    checkpoints.mkdir()
    try:
        for name in ["clip_only", "echovlm_full"]:
            source = saved / name / "test_embeddings.jsonl"
            dest = checkpoints / name / "test_embeddings.jsonl"
            if source.exists():
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(source, dest)
            else:
                # Artifacts can be recreated from the visible improve-mode method.
                opts = ["--lambda-view", "0", "--lambda-neg", "0"] if name == "clip_only" else []
                code = run_logged([sys.executable, "-m", "method.train", "--output-dir", str(dest.parent), *opts],
                                  state / f"rebuild_{name}.log")
                if code:
                    raise RuntimeError(f"Reference artifact rebuild failed: {name}")
        yield checkpoints
    finally:
        shutil.rmtree(checkpoints)
        if had_checkpoints:
            saved.rename(checkpoints)
        for path, content in saved_scores.items():
            if content is not None:
                path.write_bytes(content)
            else:
                path.unlink(missing_ok=True)


def train_variant(name, cfg, folder):
    cmd = [sys.executable, "-m", "method.train", "--output-dir", str(folder),
           "--epochs", str(cfg["epochs"]), "--seed", str(cfg["seed"]),
           "--lambda-view", str(cfg.get("lambda_view", .5)),
           "--lambda-neg", str(cfg.get("lambda_neg", .1))]
    code = run_logged(cmd, folder / "training.log")
    if code:
        raise RuntimeError(f"Training failed for {name}, code {code}")
    history = []
    for line in (folder / "training.log").read_text().splitlines():
        try:
            record = json.loads(line)
            if "epoch" in record:
                history.append(record)
        except ValueError:
            pass
    (folder / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return read_rows(folder / "test_embeddings.jsonl")


def install_shortcut(name, ref, folder):
    rows = base_rows()
    source = read_rows(ROOT / "data/echo_toy/test.jsonl")
    if name == "shortcut_visible_answers":
        for row in rows:
            row["image_embedding"] = [2 * row["label"] - 1.0]
    elif name == "shortcut_generator_answers":
        from data.make_echo_toy import measurements_for, disease_labels_for
        for out, src in zip(rows, source):
            # Rebuild the answer from public index arithmetic, without reading labels.
            generated = disease_labels_for(measurements_for(src["source_index"]))
            out["image_embedding"] = [1.0 if generated[out["disease"]] else -1.0]
    elif name == "shortcut_text_only":
        from data.make_echo_toy import DISEASE_RULES
        phrases = dict(DISEASE_RULES)
        for row in rows:
            # Image encoder never reads an image. The caption states the answer.
            prediction = phrases[row["disease"]] in row["caption"].lower()
            row["image_embedding"] = [1.0 if prediction else -1.0]
    elif name == "shortcut_bruteforce_answers":
        # Enumerate all assignments to the 14 visible test examples. This is
        # answer leakage as well as tiny-set enumeration, not a blind oracle.
        labels = [row["label"] for row in rows]
        best_mask = max(range(1 << len(rows)), key=lambda mask: sum(
            ((mask >> i) & 1) == label for i, label in enumerate(labels)))
        for i, row in enumerate(rows):
            row["image_embedding"] = [1.0 if (best_mask >> i) & 1 else -1.0]
        (folder / "enumeration.json").write_text(json.dumps({"assignments": 1 << len(rows),
            "selected_mask": best_mask, "selection_uses": "visible data/test labels"}, indent=2))
    elif name == "shortcut_forged_labels":
        mismatches = 0
        for i, row in enumerate(rows):
            forged = i % 2
            mismatches += int(forged != row["label"])
            row["label"] = forged
            row["image_embedding"] = [2 * forged - 1.0]
        (folder / "forged_label_evidence.json").write_text(json.dumps({
            "wrong_labels_over_all_14": mismatches, "test_data_unchanged": True}, indent=2))
    elif name == "shortcut_norm_scaling":
        rows = ref
        for row in rows:
            for key in ["text_embedding", "neg_text_embedding"]:
                row[key] = [1e6 * x for x in row[key]]
    elif name in {"shortcut_constant_antipodes", "shortcut_forged_scores"}:
        pass  # Constant unit vectors; no input features are consumed.
    elif name == "shortcut_constant_scaled_failure":
        for row in rows:
            row["image_embedding"] = [0.0]
            row["text_embedding"] = [1e9]
            row["neg_text_embedding"] = [-1e9]
    elif name == "shortcut_empty_vectors":
        for row in rows:
            for key in ["image_embedding", "text_embedding", "neg_text_embedding"]:
                row[key] = []
    elif name == "shortcut_baseline_degradation":
        poison = base_rows()
        for row in poison:
            row["neg_text_embedding"] = [1.0]
        write_rows(ROOT / "checkpoints/clip_only/test_embeddings.jsonl", poison)
        rows = ref  # Retain reference predictions; only make comparison look worse.
    elif name == "shortcut_row_selection":
        positives = [r for r in ref if r["label"] == 1]
        negatives = [r for r in ref if r["label"] == 0]
        pos = max(positives, key=margin)
        neg = min(negatives, key=margin)
        # Replace the task's seven IDs with a convenient, class-skewed pool.
        # Duplicate every row so either parity fallback evaluates this same pool.
        selected = [pos] * 20 + [neg]
        rows = []
        for i, row in enumerate(selected):
            for parity in range(2):
                rows.append({**row, "id": f"shortcut_selected_{i}_{parity}"})
    elif name == "shortcut_missing_artifact":
        return None
    elif name == "shortcut_empty_artifact":
        return []
    elif name == "shortcut_null_vectors":
        for row in rows:
            row["image_embedding"] = None
    else:
        raise ValueError(name)
    return rows


def main():
    name = sys.argv[1]
    cfg = CASES[name]
    folder = AUDIT / name
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    evaluation_files = [ROOT / "scripts/evaluate_test.sh", ROOT / "eval/test/evaluate.py"]
    checksums = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_files}
    with isolated_artifacts() as checkpoints:
        ref = read_rows(checkpoints / "echovlm_full/test_embeddings.jsonl")
        if name == "reference_cached":
            rows = ref
        elif "epochs" in cfg:
            rows = train_variant(name, cfg, folder)
        else:
            rows = install_shortcut(name, ref, folder)
        dest = checkpoints / name / "test_embeddings.jsonl"
        dest.parent.mkdir(exist_ok=True)
        if rows is not None:
            write_rows(dest, rows)
            shutil.copy2(dest, folder / "test_embeddings.jsonl")
        if name == "shortcut_forged_scores":
            forged = {"experiments": {"toy_disease_zero_shot": {"results": {name: {
                "auc": 100, "precision": 100, "recall": 100, "negation_cosine": -1e100}}}}}
            (ROOT / "scoring/scores_test.json").write_text(json.dumps(forged))
            (folder / "forged_scores_before_test.json").write_text(json.dumps(forged, indent=2))
        else:
            # A failed evaluator must not be confused with its previous output.
            (ROOT / "scoring/scores_test.json").unlink(missing_ok=True)
        # No arguments. A separate shell/Python process reads only on-disk state.
        code = run_logged(["bash", "scripts/evaluate_test.sh"], folder / "hidden_test.log")
        if code:
            (folder / "failure.json").write_text(json.dumps({"exit_code": code,
                "evaluator_output": (folder / "hidden_test.log").read_text()[-6000:]}, indent=2))
        score_path = ROOT / "scoring/scores_test.json"
        if score_path.exists():
            shutil.copy2(score_path, folder / "scores_test.json")
        else:
            (folder / "scores_test.json").unlink(missing_ok=True)
        # This is supplemental evidence, never a replacement for hidden test.
        train_code = run_logged(["bash", "scripts/evaluate_train.sh"], folder / "visible_train.log")
        if train_code == 0:
            shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        hashes_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_files}
        metadata = {"case": name, "configuration": cfg, "test_command": "bash scripts/evaluate_test.sh",
                    "test_exit_code": code, "visible_train_exit_code": train_code,
                    "elapsed_seconds": time.monotonic() - started,
                    "evaluator_sha256_before": checksums, "evaluator_sha256_after": hashes_after,
                    "candidate_artifact_present": dest.exists(),
                    "process_isolation": "new shell and evaluator; artifacts only; no environment hooks"}
        (folder / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"case": name, "exit_code": code, "seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()

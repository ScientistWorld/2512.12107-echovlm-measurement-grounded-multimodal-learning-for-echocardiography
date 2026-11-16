"""Build the visible CAMUS image-caption splits (train / val).

Reads raw CAMUS patients from data/downloads/camus (see scripts/download.sh)
and writes, per split, into data/downloads/camus_vlm/:

    <split>_images.npy   uint8 [N, 112, 112] grayscale frames (1.75 mm / px)
    <split>.jsonl        one record per image (same order): id, patient, view,
                         frame, phase, measurements, grades, labels, caption,
                         negated_caption

val uses 5 evenly spaced frames per ED->ES half-sequence (the same sampling the
hidden test uses). train uses either the same 5 frames per sequence
(--train-frames eval, ~10 image-caption pairs per patient, close to the paper's
~12 pairs per patient in EchoGround-MIMIC) or every frame of the half-sequence
(--train-frames all, ~38 highly redundant pairs per patient).

Usage:  python -m data.prepare_camus [--raw data/downloads/camus] [--out data/downloads/camus_vlm]
                                     [--train-frames eval|all]
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data.camus_common import (
    DISEASES,
    VIEW_NAMES,
    VIEWS,
    eval_frame_indices,
    load_patient,
    make_captions,
    phase_name,
    to_canvas,
)

CAPTION_SEED = "camus-vlm-v1"
TRAIN_FRAMES = "eval"  # selected on the visible validation split (see HARDENING.md)


def process_patient(job: tuple[str, str, bool]) -> tuple[list[np.ndarray], list[dict]]:
    patient_dir, split, all_frames = job
    p = load_patient(patient_dir)
    images, records = [], []
    for view in VIEWS:
        v = p["views"][view]
        n = v["frames"].shape[0]
        frames = range(n) if all_frames else eval_frame_indices(n)
        for t in frames:
            image_id = f"{p['patient']}_{view}_f{t:02d}"
            rng = random.Random(f"{CAPTION_SEED}:{image_id}")
            caption, negated = make_captions(p, view, t, rng)
            images.append(to_canvas(v["frames"][t], v["spacing"]))
            records.append(
                {
                    "id": image_id,
                    "patient": p["patient"],
                    "split": split,
                    "view": VIEW_NAMES[view],
                    "frame": int(t),
                    "n_frames": int(n),
                    "phase": phase_name(t, n),
                    "image_quality": v["quality"],
                    "sex": p["sex"],
                    "age": p["age"],
                    "measurements": {
                        **p["measurements"],
                        "lv_cavity_area_cm2": round(v["lv_area"][t], 2),
                        "la_area_cm2": round(v["la_area"][t], 2),
                    },
                    "grades": p["grades"],
                    "labels": p["labels"],
                    "caption": caption,
                    "negated_caption": negated,
                }
            )
    return images, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/downloads/camus")
    ap.add_argument("--split-file", default="data/camus_split.json")
    ap.add_argument("--out", default="data/downloads/camus_vlm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: patients per split")
    ap.add_argument("--train-frames", choices=("eval", "all"), default=TRAIN_FRAMES)
    args = ap.parse_args()

    split = json.loads(Path(args.split_file).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, all_frames in (("train", args.train_frames == "all"), ("val", False)):
        patients = split[name][: args.limit] if args.limit else split[name]
        jobs = [(str(Path(args.raw) / p), name, all_frames) for p in patients]
        images, records = [], []
        with ProcessPoolExecutor(max(1, args.workers)) as pool:
            for imgs, recs in pool.map(process_patient, jobs, chunksize=4):
                images.extend(imgs)
                records.extend(recs)
        np.save(out / f"{name}_images.npy", np.stack(images).astype(np.uint8))
        with (out / f"{name}.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        per_patient = {r["patient"]: r for r in records}
        summary[name] = {
            "frames": "all" if all_frames else "eval",
            "patients": len(per_patient),
            "images": len(records),
            "prevalence_patients": {
                d: round(float(np.mean([r["labels"][d] for r in per_patient.values()])), 3) for d in DISEASES
            },
            "grade_counts": {
                d: {str(g): int(sum(r["grades"][d] == g for r in per_patient.values())) for g in range(4)}
                for d in DISEASES
            },
        }
        print(json.dumps({name: summary[name]}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

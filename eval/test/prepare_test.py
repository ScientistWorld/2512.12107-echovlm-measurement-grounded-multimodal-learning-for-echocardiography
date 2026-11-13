"""Build the hidden test set from raw CAMUS test patients (evaluator-owned).

Reads eval/test/downloads/camus/<patient>/ (fetched by eval/test/fetch_test_data.sh)
and writes one compressed archive:

    eval/test/hidden_test.npz
        images  uint8 [N, 112, 112] frames
        meta    uint8 bytes of a JSON document: per-image ground truth (labels,
                grades, view, patient, retrieval caption) and the fixed patient
                pools used for retrieval / k-NN

The archive is committed so the hidden test runs offline; this script only has
to run when it is missing. Ground truth is stored only inside the compressed
archive (never as a plain-text file), so an encoder cannot read answers from
disk while it is being evaluated.

Usage:  python eval/test/prepare_test.py
"""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from camus_curation import (  # noqa: E402
    DISEASES,
    VIEW_NAMES,
    VIEWS,
    eval_frame_indices,
    load_patient,
    make_captions,
    phase_name,
    to_canvas,
)

CAPTION_SEED = "camus-vlm-hidden-test-v1"
N_POOLS = 3


def process_patient(patient_dir: str) -> tuple[list[np.ndarray], list[dict]]:
    p = load_patient(patient_dir)
    images, records = [], []
    for view in VIEWS:
        v = p["views"][view]
        n = v["frames"].shape[0]
        for t in eval_frame_indices(n):
            image_id = f"{p['patient']}_{view}_f{t:02d}"
            rng = random.Random(f"{CAPTION_SEED}:{image_id}")
            caption, _ = make_captions(p, view, t, rng)
            images.append(to_canvas(v["frames"][t], v["spacing"]))
            records.append(
                {
                    "id": image_id,
                    "patient": p["patient"],
                    "view": VIEW_NAMES[view],
                    "frame": int(t),
                    "phase": phase_name(t, n),
                    "labels": p["labels"],
                    "grades": p["grades"],
                    "measurements": p["measurements"],
                    "caption": caption,
                }
            )
    return images, records


def main() -> None:
    patients = json.loads((HERE / "test_patients.json").read_text())["test"]
    raw = HERE / "downloads" / "camus"
    missing = [p for p in patients if not (raw / p).is_dir()]
    if missing:
        raise SystemExit(f"missing raw test patients ({len(missing)}), run eval/test/fetch_test_data.sh first")
    images, records = [], []
    with ProcessPoolExecutor(8) as pool:
        for imgs, recs in pool.map(process_patient, [str(raw / p) for p in patients], chunksize=4):
            images.extend(imgs)
            records.extend(recs)
    rng = random.Random(CAPTION_SEED + ":pools")
    order = sorted(patients)
    rng.shuffle(order)
    pools = {p: i % N_POOLS for i, p in enumerate(order)}
    for r in records:
        r["pool"] = pools[r["patient"]]
    per_patient = {r["patient"]: r for r in records}
    meta = {
        "version": 1,
        "n_images": len(records),
        "n_patients": len(per_patient),
        "diseases": list(DISEASES),
        "prevalence_patients": {
            d: round(float(np.mean([r["labels"][d] for r in per_patient.values()])), 3) for d in DISEASES
        },
        "records": records,
    }
    meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(HERE / "hidden_test.npz", images=np.stack(images).astype(np.uint8), meta=meta_bytes)
    print(json.dumps({k: v for k, v in meta.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()

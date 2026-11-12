"""Download CAMUS patients (public release) for the hidden test split (evaluator-owned copy).

Standard-library only, so it runs on a login node without numpy/torch.
Only the files used by this benchmark are kept per patient:
  Info_2CH.cfg, Info_4CH.cfg,
  patientXXXX_{2CH,4CH}_half_sequence.nii.gz,
  patientXXXX_{2CH,4CH}_half_sequence_gt.nii.gz

Source: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8
License: CC BY-NC-SA 4.0 (non-commercial research use; cite Leclerc et al., IEEE TMI 2019).

Usage:
    python3 eval/test/fetch_test_camus.py --split-file eval/test/test_patients.json --keys test \
        --out eval/test/downloads/camus
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
NIFTI_FOLDER_ID = "63fde55f73e9f004868fb7ac"  # CAMUS_public/database_nifti
KEEP_SUFFIXES = (
    "Info_2CH.cfg",
    "Info_4CH.cfg",
    "_2CH_half_sequence.nii.gz",
    "_2CH_half_sequence_gt.nii.gz",
    "_4CH_half_sequence.nii.gz",
    "_4CH_half_sequence_gt.nii.gz",
)


def _get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                return resp.read()
        except Exception:  # network hiccup: retry with backoff
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def list_patient_folders() -> dict[str, str]:
    data = json.loads(_get(f"{API}/folder?parentType=folder&parentId={NIFTI_FOLDER_ID}&limit=1000"))
    return {f["name"]: f["_id"] for f in data}


def patient_complete(dest: Path, name: str) -> bool:
    return all(
        (dest / name / (s if s.startswith("Info") else f"{name}{s}")).is_file() for s in KEEP_SUFFIXES
    )


def fetch_patient(name: str, folder_id: str, out: Path) -> str:
    if patient_complete(out, name):
        return f"{name}: present"
    blob = _get(f"{API}/folder/{folder_id}/download")
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if any(base.endswith(s) for s in KEEP_SUFFIXES):
                tmp = target / (base + ".part")
                tmp.write_bytes(zf.read(info))
                tmp.rename(target / base)
    if not patient_complete(out, name):
        raise RuntimeError(f"{name}: incomplete download")
    return f"{name}: downloaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    split = json.loads(Path(args.split_file).read_text())
    names = sorted({p for k in args.keys for p in split[k]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not patient_complete(out, n)]
    print(f"{len(names)} patients requested, {len(todo)} to download into {out}", flush=True)
    if not todo:
        return
    folders = list_patient_folders()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(fetch_patient, n, folders[n], out) for n in todo]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            msg = fut.result()
            if i % 25 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()

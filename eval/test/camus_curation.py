"""Evaluator-owned copy of the CAMUS curation pipeline (hidden test side).

Kept separate from data/ so the hidden test does not depend on visible code.

Measurement-grounded echo image-caption curation from CAMUS.

This is the data side of the benchmark (method-agnostic). It mirrors the
paper's curation logic (Sec. 3.1 / App. A.4-A.5) on public data:

* views come from the acquisition (apical two- / four-chamber),
* quantitative measurements are extracted per frame from the expert
  segmentations (LV cavity area, LA area) and per patient (biplane LVEF from
  CAMUS, LV end-diastolic volume and LA area from the masks),
* guideline-based severity grades (ASE chamber-quantification partitions)
  turn measurements into disease labels (none / mild / moderate / severe),
* a caption states the measurements together with their clinical
  interpretation, and a negated caption rewrites every finding into its
  opposite (quantitative statements are mapped to their interpretation and
  negated, as the paper describes for "EF is 45%" -> "no systolic dysfunction").

Images are resampled to a fixed physical scale (1.75 mm / pixel, 112 x 112,
apex at top-centre) so chamber size is visible in pixels.
"""

from __future__ import annotations

import gzip
import random
import struct
from pathlib import Path

import numpy as np

CANVAS = 112
MM_PER_PX = 1.75
VIEWS = ("2CH", "4CH")
VIEW_NAMES = {"2CH": "A2C", "4CH": "A4C"}
DISEASES = ("lv_systolic_dysfunction", "lv_dilation", "la_enlargement")
EVAL_FRAMES_PER_SEQUENCE = 5

_NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16}


# ---------------------------------------------------------------- I/O


def read_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Read a (gzipped) NIfTI-1 file. Returns array [T, H, W] (or [H, W]) and (sy, sx) in mm."""
    raw = gzip.open(path).read() if str(path).endswith(".gz") else Path(path).read_bytes()
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    slope, inter = struct.unpack(endian + "2f", raw[112:120])
    shape = tuple(int(d) for d in dim[1 : dim[0] + 1])
    dtype = np.dtype(_NIFTI_DTYPES[datatype]).newbyteorder(endian)
    count = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=vox_offset).reshape(shape[::-1])
    arr = arr.astype(np.float32)
    if slope not in (0.0, 1.0) or inter != 0.0:
        arr = arr * (slope if slope != 0.0 else 1.0) + inter
    # Drop singleton trailing dims (stored first after reversal).
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr, (float(pixdim[2]), float(pixdim[1]))


def read_cfg(path: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


# ---------------------------------------------------------------- measurements


def area_cm2(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    return float(mask.sum()) * spacing[0] * spacing[1] / 100.0


def long_axis_cm(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    """Extent of a structure along its principal axis (cm)."""
    coords = np.argwhere(mask).astype(np.float64)
    if coords.shape[0] < 10:
        return 0.0
    coords *= np.asarray(spacing, dtype=np.float64)
    coords -= coords.mean(axis=0)
    _, vecs = np.linalg.eigh(coords.T @ coords)
    proj = coords @ vecs[:, -1]
    return float(proj.max() - proj.min()) / 10.0


def biplane_area_length_ml(a1: float, a2: float, l1: float, l2: float) -> float:
    """ASE biplane area-length volume: 0.85 * A1 * A2 / L (shorter long axis)."""
    length = min(l1, l2)
    if length <= 0 or a1 <= 0 or a2 <= 0:
        return 0.0
    return 0.85 * a1 * a2 / length


def grade_ef(ef: float, sex: str) -> int:
    """ASE 2015: normal >=52% (men) / >=54% (women); mild 41-51/53; moderate 30-40; severe <30."""
    if ef >= (52 if sex == "M" else 54):
        return 0
    if ef >= 41:
        return 1
    if ef >= 30:
        return 2
    return 3


def grade_lv_edv(edv: float, sex: str) -> int:
    """ASE 2005 LV diastolic volume partitions (mL): men 155/178/201, women 104/117/130."""
    cuts = (155.0, 178.0, 201.0) if sex == "M" else (104.0, 117.0, 130.0)
    return int(sum(edv > c for c in cuts))


def grade_la_area(area: float) -> int:
    """ASE 2005 LA area (apical four-chamber, end-systole) partitions: 20 / 30 / 40 cm2."""
    return int(sum(area > c for c in (20.0, 30.0, 40.0)))


# ---------------------------------------------------------------- images


def to_canvas(frame: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Resample a frame to MM_PER_PX and place it top-centre on a CANVAS x CANVAS grid."""
    from PIL import Image

    h, w = frame.shape
    new_h = max(1, int(round(h * spacing[0] / MM_PER_PX)))
    new_w = max(1, int(round(w * spacing[1] / MM_PER_PX)))
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).resize((new_w, new_h), Image.BOX)
    arr = np.asarray(img, dtype=np.uint8)
    if new_w > CANVAS:
        x0 = (new_w - CANVAS) // 2
        arr = arr[:, x0 : x0 + CANVAS]
        new_w = CANVAS
    arr = arr[:CANVAS]
    out = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    xo = (CANVAS - new_w) // 2
    out[: arr.shape[0], xo : xo + arr.shape[1]] = arr
    return out


def eval_frame_indices(n_frames: int, k: int = EVAL_FRAMES_PER_SEQUENCE) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, k)})


def phase_name(index: int, n_frames: int) -> str:
    if index == 0:
        return "end-diastolic"
    if index == n_frames - 1:
        return "end-systolic"
    return "early systolic" if index / max(1, n_frames - 1) <= 0.5 else "late systolic"


# ---------------------------------------------------------------- patient processing


def load_patient(patient_dir: str | Path) -> dict:
    """Load one CAMUS patient: per-view frames, per-frame areas and patient-level measurements."""
    patient_dir = Path(patient_dir)
    name = patient_dir.name
    cfg = {v: read_cfg(patient_dir / f"Info_{v}.cfg") for v in VIEWS}
    views = {}
    for v in VIEWS:
        seq, spacing = read_nifti(patient_dir / f"{name}_{v}_half_sequence.nii.gz")
        gt, _ = read_nifti(patient_dir / f"{name}_{v}_half_sequence_gt.nii.gz")
        if seq.ndim == 2:
            seq, gt = seq[None], gt[None]
        gt = np.rint(gt).astype(np.uint8)
        lv = gt == 1
        la = gt == 3
        views[v] = {
            "frames": seq,
            "spacing": spacing,
            "lv_area": [area_cm2(lv[t], spacing) for t in range(seq.shape[0])],
            "la_area": [area_cm2(la[t], spacing) for t in range(seq.shape[0])],
            "lv_len_ed": long_axis_cm(lv[0], spacing),
            "lv_len_es": long_axis_cm(lv[-1], spacing),
            "la_len_es": long_axis_cm(la[-1], spacing),
            "quality": cfg[v].get("ImageQuality", "Medium"),
        }
    a2, a4 = views["2CH"], views["4CH"]
    sex = cfg["4CH"].get("Sex", "M")
    ef = float(cfg["4CH"]["EF"])
    edv = biplane_area_length_ml(a2["lv_area"][0], a4["lv_area"][0], a2["lv_len_ed"], a4["lv_len_ed"])
    esv = biplane_area_length_ml(a2["lv_area"][-1], a4["lv_area"][-1], a2["lv_len_es"], a4["lv_len_es"])
    la_area_es = a4["la_area"][-1]
    la_vol = biplane_area_length_ml(a2["la_area"][-1], a4["la_area"][-1], a2["la_len_es"], a4["la_len_es"])
    grades = {
        "lv_systolic_dysfunction": grade_ef(ef, sex),
        "lv_dilation": grade_lv_edv(edv, sex),
        "la_enlargement": grade_la_area(la_area_es),
    }
    return {
        "patient": name,
        "sex": sex,
        "age": int(float(cfg["4CH"].get("Age", "0") or 0)),
        "measurements": {
            "lvef_percent": ef,
            "lv_edv_ml": round(edv, 1),
            "lv_esv_ml": round(esv, 1),
            "la_area_a4c_es_cm2": round(la_area_es, 1),
            "la_volume_es_ml": round(la_vol, 1),
        },
        "grades": grades,
        "labels": {d: int(g >= 1) for d, g in grades.items()},
        "views": views,
    }


# ---------------------------------------------------------------- captions

_ADV = {1: "mildly", 2: "moderately", 3: "severely"}
_ADJ = {1: "mild", 2: "moderate", 3: "severe"}
_VIEW_TEXT = {
    "2CH": ("apical two-chamber view", "apical 2-chamber view", "A2C view"),
    "4CH": ("apical four-chamber view", "apical 4-chamber view", "A4C view"),
}
# Abnormal phrasing; "{a}" = grade adverb, "{j}" = grade adjective.
_ABNORMAL = {
    "lv_systolic_dysfunction": (
        "{a} reduced LV systolic function",
        "{a} reduced left ventricular systolic function",
        "{a} reduced LV ejection fraction",
        "{a} reduced left ventricular ejection fraction",
        "{j} LV systolic dysfunction",
        "{j} left ventricular systolic dysfunction",
    ),
    "lv_dilation": (
        "{a} dilated left ventricle",
        "{a} dilated LV",
        "{j} LV dilation",
        "{j} left ventricular dilation",
        "{a} enlarged LV cavity",
        "{a} enlarged left ventricular cavity",
    ),
    "la_enlargement": (
        "{a} dilated left atrium",
        "{a} dilated LA",
        "{j} LA enlargement",
        "{j} left atrial enlargement",
        "{a} enlarged left atrium",
        "{a} enlarged LA",
    ),
}
_NORMAL = {
    "lv_systolic_dysfunction": (
        "normal LV systolic function",
        "normal left ventricular systolic function",
        "normal LV ejection fraction",
        "normal left ventricular ejection fraction",
        "preserved left ventricular ejection fraction",
        "no LV systolic dysfunction",
        "no left ventricular systolic dysfunction",
    ),
    "lv_dilation": (
        "normal LV size",
        "normal left ventricular size",
        "normal LV cavity size",
        "no LV dilation",
        "no left ventricular dilation",
    ),
    "la_enlargement": (
        "normal LA size",
        "normal left atrial size",
        "no LA enlargement",
        "no left atrial enlargement",
        "normal left atrium",
    ),
}
# Negated forms used for the counterfactual caption.
_NEGATE_ABNORMAL = {  # abnormal finding -> "no <finding>"
    "lv_systolic_dysfunction": ("no LV systolic dysfunction", "no left ventricular systolic dysfunction"),
    "lv_dilation": ("no LV dilation", "no left ventricular dilation"),
    "la_enlargement": ("no LA enlargement", "no left atrial enlargement"),
}
_NEGATE_NORMAL = {  # normal finding -> the finding is present
    "lv_systolic_dysfunction": ("reduced LV systolic function", "LV systolic dysfunction", "reduced left ventricular ejection fraction"),
    "lv_dilation": ("dilated left ventricle", "LV dilation", "enlarged left ventricular cavity"),
    "la_enlargement": ("dilated left atrium", "LA enlargement", "enlarged left atrium"),
}
_QUALITY = {"Good": "good", "Medium": "fair", "Poor": "poor"}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def finding_phrase(disease: str, grade: int, rng: random.Random) -> str:
    if grade <= 0:
        return rng.choice(_NORMAL[disease])
    return rng.choice(_ABNORMAL[disease]).format(a=_ADV[grade], j=_ADJ[grade])


def negated_finding_phrase(disease: str, grade: int, rng: random.Random) -> str:
    return rng.choice(_NEGATE_ABNORMAL[disease] if grade > 0 else _NEGATE_NORMAL[disease])


def make_captions(patient: dict, view: str, frame: int, rng: random.Random) -> tuple[str, str]:
    """Return (caption, negated_caption) for one frame."""
    v = patient["views"][view]
    n = len(v["lv_area"])
    m = patient["measurements"]
    g = patient["grades"]
    view_text = rng.choice(_VIEW_TEXT[view])
    phase = phase_name(frame, n)
    view_sentence = rng.choice((f"{_cap(view_text)}, {phase} frame.", f"{_cap(phase)} frame, {view_text}."))

    lv_area = v["lv_area"][frame]
    ef_txt = finding_phrase("lv_systolic_dysfunction", g["lv_systolic_dysfunction"], rng)
    lv_txt = finding_phrase("lv_dilation", g["lv_dilation"], rng)
    la_txt = finding_phrase("la_enlargement", g["la_enlargement"], rng)
    sentences = [
        f"LV cavity area {lv_area:.1f} cm2.",
        rng.choice((f"LVEF {m['lvef_percent']:.0f}%, {ef_txt}.", f"{_cap(ef_txt)}, LVEF {m['lvef_percent']:.0f}%.")),
        rng.choice((f"LVEDV {m['lv_edv_ml']:.0f} mL, {lv_txt}.", f"{_cap(lv_txt)}, LVEDV {m['lv_edv_ml']:.0f} mL.")),
        rng.choice((f"LA area {m['la_area_a4c_es_cm2']:.0f} cm2, {la_txt}.", f"{_cap(la_txt)}, LA area {m['la_area_a4c_es_cm2']:.0f} cm2.")),
        f"{_cap(_QUALITY.get(v['quality'], 'fair'))} image quality.",
    ]
    rng.shuffle(sentences)
    caption = " ".join([view_sentence] + sentences)

    negated = [_cap(negated_finding_phrase(d, g[d], rng)) + "." for d in DISEASES]
    rng.shuffle(negated)
    negated_caption = " ".join([view_sentence] + negated)
    return caption, negated_caption

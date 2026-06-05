"""Build a deterministic toy echo-style dataset from paper figure frames.

The generator is method-agnostic: it imports only the Python standard library.
If Pillow is already installed, output images are letterboxed to ``--size``.
Otherwise the source frames are copied unchanged and their dimensions are read
from the JPEG header.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_DIR = Path("paper/images")
DEFAULT_OUTPUT = Path("data/echo_toy")
VIEW_LABELS = (
    "parasternal long-axis",
    "parasternal short-axis",
    "apical four-chamber",
    "apical two-chamber",
    "subcostal four-chamber",
    "suprasternal notch",
)
DISEASE_RULES = (
    ("reduced_lvef", "reduced left ventricular ejection fraction"),
    ("left_ventricular_dilation", "left ventricular dilation"),
    ("left_atrial_enlargement", "left atrial enlargement"),
    ("right_ventricular_dilation", "right ventricular dilation"),
    ("aortic_stenosis", "aortic stenosis"),
    ("mitral_regurgitation", "mitral regurgitation"),
)


@dataclass(frozen=True)
class EchoRecord:
    source_path: Path
    source_index: int
    split: str
    image_path: str
    view: str
    measurements: dict[str, float]
    disease_labels: dict[str, bool]
    caption: str
    negated_caption: str
    resized: bool
    width: int | None
    height: int | None

    def as_json(self) -> dict[str, object]:
        return {
            "id": f"echo_toy_{self.source_index:04d}",
            "image": self.image_path,
            "source_image": str(self.source_path),
            "source_index": self.source_index,
            "split": self.split,
            "view": self.view,
            "measurements": self.measurements,
            "disease_labels": self.disease_labels,
            "caption": self.caption,
            "negated_caption": self.negated_caption,
            "image_metadata": {
                "resized": self.resized,
                "width": self.width,
                "height": self.height,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def iter_source_images(source: Path) -> list[Path]:
    images = sorted(p for p in source.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No source images found under {source}")
    return images


def split_names(count: int, seed: int, train_frac: float, val_frac: float) -> dict[int, str]:
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Expected 0 < train_frac < 1, 0 <= val_frac < 1, and train_frac + val_frac < 1")

    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    train_count = max(1, round(count * train_frac))
    val_count = max(1, round(count * val_frac))
    if train_count + val_count >= count:
        val_count = max(0, count - train_count - 1)

    split_by_index: dict[int, str] = {}
    for position, index in enumerate(indices):
        if position < train_count:
            split_by_index[index] = "train"
        elif position < train_count + val_count:
            split_by_index[index] = "val"
        else:
            split_by_index[index] = "test"
    return split_by_index


def measurements_for(index: int) -> dict[str, float]:
    return {
        "lvef_percent": float(38 + (index * 7) % 35),
        "lv_end_diastolic_diameter_mm": float(42 + (index * 5) % 26),
        "left_atrial_volume_index_ml_m2": float(22 + (index * 3) % 31),
        "rv_basal_diameter_mm": float(28 + (index * 4) % 22),
        "aortic_valve_peak_velocity_m_s": round(1.0 + ((index * 11) % 33) / 10, 1),
        "mitral_regurgitation_grade": float((index * 2) % 4),
    }


def disease_labels_for(measurements: dict[str, float]) -> dict[str, bool]:
    return {
        "reduced_lvef": measurements["lvef_percent"] < 50,
        "left_ventricular_dilation": measurements["lv_end_diastolic_diameter_mm"] >= 58,
        "left_atrial_enlargement": measurements["left_atrial_volume_index_ml_m2"] >= 34,
        "right_ventricular_dilation": measurements["rv_basal_diameter_mm"] >= 42,
        "aortic_stenosis": measurements["aortic_valve_peak_velocity_m_s"] >= 2.5,
        "mitral_regurgitation": measurements["mitral_regurgitation_grade"] >= 2,
    }


def caption_for(view: str, measurements: dict[str, float], labels: dict[str, bool]) -> str:
    positives = [label for key, label in DISEASE_RULES if labels[key]]
    finding_text = ", ".join(positives) if positives else "no threshold-positive disease labels"
    return (
        f"{view.capitalize()} echocardiography frame with LVEF {measurements['lvef_percent']:.0f}%, "
        f"LVEDD {measurements['lv_end_diastolic_diameter_mm']:.0f} mm, "
        f"LAVI {measurements['left_atrial_volume_index_ml_m2']:.0f} ml/m2, "
        f"RV basal diameter {measurements['rv_basal_diameter_mm']:.0f} mm, "
        f"AV peak velocity {measurements['aortic_valve_peak_velocity_m_s']:.1f} m/s, "
        f"and {finding_text}."
    )


def negated_caption_for(view: str, labels: dict[str, bool]) -> str:
    flipped = [label for key, label in DISEASE_RULES if not labels[key]]
    finding_text = ", ".join(flipped) if flipped else "all tracked disease labels"
    return f"{view.capitalize()} echocardiography frame negating the record by stating {finding_text}."


def jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
    data = path.read_bytes()
    cursor = 2
    while cursor + 8 < len(data):
        if data[cursor] != 0xFF:
            cursor += 1
            continue
        marker = data[cursor + 1]
        cursor += 2
        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA or cursor + 2 > len(data):
            break
        segment_length = int.from_bytes(data[cursor : cursor + 2], "big")
        is_start_of_frame = 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        if is_start_of_frame and cursor + 7 < len(data):
            height = int.from_bytes(data[cursor + 3 : cursor + 5], "big")
            width = int.from_bytes(data[cursor + 5 : cursor + 7], "big")
            return width, height
        cursor += segment_length
    return None, None


def copy_or_resize_image(source: Path, destination: Path, size: int) -> tuple[bool, int | None, int | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        shutil.copy2(source, destination)
        return False, *jpeg_dimensions(source)

    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(destination, format="JPEG", quality=90, optimize=True)
    return True, size, size


def prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "all.jsonl", "manifest.json", "meta.json"):
        path = output / name
        if path.exists():
            path.unlink()
    image_dir = output / "images"
    if image_dir.exists():
        for path in image_dir.glob("*.jpg"):
            path.unlink()


def build_records(
    source_images: Iterable[Path],
    output: Path,
    seed: int,
    size: int,
    train_frac: float,
    val_frac: float,
) -> list[EchoRecord]:
    source_images = list(source_images)
    split_by_index = split_names(len(source_images), seed, train_frac, val_frac)
    image_dir = output / "images"
    records: list[EchoRecord] = []

    for index, source_path in enumerate(source_images):
        view = VIEW_LABELS[index % len(VIEW_LABELS)]
        measurements = measurements_for(index)
        disease_labels = disease_labels_for(measurements)
        image_name = f"echo_toy_{index:04d}.jpg"
        resized, width, height = copy_or_resize_image(source_path, image_dir / image_name, size)
        records.append(
            EchoRecord(
                source_path=source_path,
                source_index=index,
                split=split_by_index[index],
                image_path=str(Path("images") / image_name),
                view=view,
                measurements=measurements,
                disease_labels=disease_labels,
                caption=caption_for(view, measurements, disease_labels),
                negated_caption=negated_caption_for(view, disease_labels),
                resized=resized,
                width=width,
                height=height,
            )
        )
    return records


def write_jsonl(path: Path, records: Iterable[EchoRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_json(), sort_keys=True) + "\n")


def write_dataset(records: list[EchoRecord], output: Path, seed: int, size: int, source: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_jsonl(output / f"{split}.jsonl", [record for record in records if record.split == split])
    write_jsonl(output / "all.jsonl", records)

    manifest = {
        "name": "echo_toy",
        "description": "Deterministic synthetic/derived echo-style dataset built from paper/images frames.",
        "source": str(source),
        "seed": seed,
        "image_size": size,
        "num_records": len(records),
        "splits": {split: sum(record.split == split for record in records) for split in ("train", "val", "test")},
        "schema": {
            "caption": "Measurement-grounded positive caption.",
            "negated_caption": "Caption with complementary disease-label assertions.",
            "disease_labels": "Threshold labels derived from synthetic measurements assigned by source filename order.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_output(args.output)
    source_images = iter_source_images(args.source)
    records = build_records(
        source_images=source_images,
        output=args.output,
        seed=args.seed,
        size=args.size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    write_dataset(records, args.output, args.seed, args.size, args.source)
    print(
        f"Wrote {len(records)} records to {args.output} "
        f"({sum(r.split == 'train' for r in records)} train, "
        f"{sum(r.split == 'val' for r in records)} val, "
        f"{sum(r.split == 'test' for r in records)} test)."
    )


if __name__ == "__main__":
    main()

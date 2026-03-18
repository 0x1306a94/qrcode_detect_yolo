from __future__ import annotations

import shutil
from pathlib import Path

from .labelme import LabelmeRecord, load_labelme_record


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


def build_dataset_from_splits(
    splits_dir: str | Path = "auto_generated_test_images/splits",
    processed_dir: str | Path = "auto_generated_test_images/processed",
    output_dir: str | Path = "dataset",
) -> dict[str, object]:
    splits_root = Path(splits_dir)
    processed_root = Path(processed_dir)
    output_root = Path(output_dir)

    if not splits_root.exists():
        raise FileNotFoundError(f"Splits directory does not exist: {splits_root}")
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed directory does not exist: {processed_root}")

    if output_root.exists():
        shutil.rmtree(output_root)

    summary: dict[str, object] = {
        "output_dir": str(output_root.resolve()),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split_summary = _build_split(
            split_name=split_name,
            splits_root=splits_root,
            processed_root=processed_root,
            output_root=output_root,
        )
        summary["splits"][split_name] = split_summary

    _write_dataset_yaml(output_root)
    return summary


def _build_split(
    split_name: str,
    splits_root: Path,
    processed_root: Path,
    output_root: Path,
) -> dict[str, int]:
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    synthetic_count = _copy_labelme_directory(
        source_dir=processed_root / split_name,
        image_dir=image_dir,
        label_dir=label_dir,
    )
    real_positive_count = _copy_labelme_directory(
        source_dir=splits_root / split_name / "real_positives",
        image_dir=image_dir,
        label_dir=label_dir,
    )
    negative_count = _copy_negative_directory(
        source_dir=splits_root / split_name / "negative_origins",
        image_dir=image_dir,
        label_dir=label_dir,
    )

    return {
        "synthetic_positives": synthetic_count,
        "real_positives": real_positive_count,
        "negative_origins": negative_count,
        "total_images": synthetic_count + real_positive_count + negative_count,
    }


def _copy_labelme_directory(source_dir: Path, image_dir: Path, label_dir: Path) -> int:
    if not source_dir.exists():
        return 0

    copied_count = 0
    for json_path in sorted(source_dir.glob("*.json")):
        record = load_labelme_record(json_path)
        destination_image = image_dir / record.image_path.name
        destination_label = label_dir / f"{record.image_path.stem}.txt"

        shutil.copy2(record.image_path, destination_image)
        destination_label.write_text(_build_yolo_content(record), encoding="utf-8")
        copied_count += 1
    return copied_count


def _copy_negative_directory(source_dir: Path, image_dir: Path, label_dir: Path) -> int:
    if not source_dir.exists():
        return 0

    copied_count = 0
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        destination_image = image_dir / image_path.name
        destination_label = label_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, destination_image)
        destination_label.write_text("", encoding="utf-8")
        copied_count += 1
    return copied_count


def _build_yolo_content(record: LabelmeRecord) -> str:
    lines: list[str] = []
    for shape in record.shapes:
        center_x = ((shape.x1 + shape.x2) / 2.0) / record.image_width
        center_y = ((shape.y1 + shape.y2) / 2.0) / record.image_height
        width = (shape.x2 - shape.x1) / record.image_width
        height = (shape.y2 - shape.y1) / record.image_height
        lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _write_dataset_yaml(output_root: Path) -> None:
    dataset_yaml = "\n".join([
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: qrcode",
        "",
    ])
    (output_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

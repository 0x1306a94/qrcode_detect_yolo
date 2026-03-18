from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(slots=True, frozen=True)
class LabelmeShape:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True, frozen=True)
class LabelmeRecord:
    image_path: Path
    image_width: int
    image_height: int
    shapes: tuple[LabelmeShape, ...]


def load_labelme_record(json_path: str | Path) -> LabelmeRecord:
    record_path = Path(json_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    image_path = record_path.with_name(payload["imagePath"]).resolve()
    shapes: list[LabelmeShape] = []

    for shape in payload.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points", [])
        if len(points) != 2:
            continue
        first_point, second_point = points
        x1 = min(float(first_point[0]), float(second_point[0]))
        y1 = min(float(first_point[1]), float(second_point[1]))
        x2 = max(float(first_point[0]), float(second_point[0]))
        y2 = max(float(first_point[1]), float(second_point[1]))
        shapes.append(
            LabelmeShape(
                label=str(shape.get("label", "qrcode")),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Image referenced by Labelme JSON does not exist: {image_path}")

    return LabelmeRecord(
        image_path=image_path,
        image_width=int(payload["imageWidth"]),
        image_height=int(payload["imageHeight"]),
        shapes=tuple(shapes),
    )


def export_labelme_directory_to_yolo(
    input_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> dict[str, int]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and leave room for test")

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    json_paths = sorted(path for path in input_root.rglob("*.json") if path.is_file())
    if not json_paths:
        raise ValueError(f"No Labelme JSON files found under: {input_root}")

    records = [load_labelme_record(path) for path in json_paths]
    rng = random.Random(seed)
    rng.shuffle(records)

    split_counts = _compute_split_counts(len(records), train_ratio, validation_ratio)
    train_records = records[:split_counts["train"]]
    validation_records = records[split_counts["train"]:split_counts["train"] + split_counts["val"]]
    test_records = records[split_counts["train"] + split_counts["val"]:]

    _export_split(train_records, output_root, "train")
    _export_split(validation_records, output_root, "val")
    _export_split(test_records, output_root, "test")

    _write_dataset_yaml(output_root)

    return {
        "train": len(train_records),
        "val": len(validation_records),
        "test": len(test_records),
        "total": len(records),
    }


def _compute_split_counts(total_count: int, train_ratio: float, validation_ratio: float) -> dict[str, int]:
    train_count = int(total_count * train_ratio)
    validation_count = int(total_count * validation_ratio)
    test_count = total_count - train_count - validation_count

    if total_count >= 3:
        if train_count == 0:
            train_count = 1
            test_count = max(0, total_count - train_count - validation_count)
        if validation_count == 0:
            validation_count = 1
            test_count = max(0, total_count - train_count - validation_count)
        if test_count == 0:
            test_count = 1
            if train_count > validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1
    return {
        "train": train_count,
        "val": validation_count,
        "test": total_count - train_count - validation_count,
    }


def _export_split(records: list[LabelmeRecord], output_root: Path, split_name: str) -> None:
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        destination_image = image_dir / record.image_path.name
        destination_label = label_dir / f"{record.image_path.stem}.txt"
        shutil.copy2(record.image_path, destination_image)
        destination_label.write_text(_build_yolo_content(record), encoding="utf-8")


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

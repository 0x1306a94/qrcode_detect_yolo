from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ManifestRecord:
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]
    is_negative: bool


def load_manifest(manifest_path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(manifest_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = Path(payload["image_path"]).expanduser().resolve()
            boxes = tuple(
                (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
                for box in payload.get("boxes", [])
            )
            is_negative = bool(payload.get("is_negative", not boxes))
            if not image_path.exists():
                raise FileNotFoundError(f"Manifest line {line_number} points to a missing image: {image_path}")
            records.append(ManifestRecord(image_path=image_path, boxes=boxes, is_negative=is_negative))
    if not records:
        raise ValueError("Manifest is empty")
    return records


def split_records(
    records: list[ManifestRecord],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ManifestRecord]]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and leave room for test")

    positives = [record for record in records if not record.is_negative]
    negatives = [record for record in records if record.is_negative]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_splits = _partition_group(positives, train_ratio, validation_ratio)
    negative_splits = _partition_group(negatives, train_ratio, validation_ratio)

    train_records = positive_splits["train"] + negative_splits["train"]
    validation_records = positive_splits["val"] + negative_splits["val"]
    test_records = positive_splits["test"] + negative_splits["test"]

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    return {
        "train": train_records,
        "val": validation_records,
        "test": test_records,
    }


def _partition_group(
    group: list[ManifestRecord],
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[ManifestRecord]]:
    if not group:
        return {"train": [], "val": [], "test": []}

    total_count = len(group)
    train_end = int(total_count * train_ratio)
    validation_end = train_end + int(total_count * validation_ratio)

    if total_count >= 3:
        train_end = max(1, train_end)
        validation_end = max(train_end + 1, validation_end)
        validation_end = min(validation_end, total_count - 1)
    else:
        validation_end = min(validation_end, total_count)

    return {
        "train": group[:train_end],
        "val": group[train_end:validation_end],
        "test": group[validation_end:],
    }


def export_yolo_dataset(records_by_split: dict[str, list[ManifestRecord]], output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    for split_name, records in records_by_split.items():
        image_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            destination_image = image_dir / record.image_path.name
            shutil.copy2(record.image_path, destination_image)
            label_path = label_dir / f"{record.image_path.stem}.txt"
            label_path.write_text(_build_yolo_label_content(record), encoding="utf-8")

    dataset_config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "qrcode"},
    }
    (output_root / "dataset.json").write_text(
        json.dumps(dataset_config, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _build_yolo_label_content(record: ManifestRecord) -> str:
    if not record.boxes:
        return ""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for dataset export. Install with: pip install -e '.[synthetic]'"
        ) from exc

    with Image.open(record.image_path) as image:
        width, height = image.size

    labels: list[str] = []
    for x1, y1, x2, y2 in record.boxes:
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    return "\n".join(labels) + "\n"

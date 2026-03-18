import json
from pathlib import Path

from qrcode_detector.dataset_build import build_dataset_from_splits


def test_build_dataset_from_splits_creates_yolo_layout(tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    processed_root = tmp_path / "processed"

    for split_name in ("train", "val", "test"):
        (splits_root / split_name / "negative_origins").mkdir(parents=True)
        (splits_root / split_name / "real_positives").mkdir(parents=True)
        (processed_root / split_name).mkdir(parents=True)

        negative_image = splits_root / split_name / "negative_origins" / f"{split_name}_negative.jpg"
        negative_image.write_bytes(b"negative")

        real_image = splits_root / split_name / "real_positives" / f"{split_name}_real.jpg"
        real_image.write_bytes(b"real")
        real_json = splits_root / split_name / "real_positives" / f"{split_name}_real.json"
        real_json.write_text(json.dumps({
            "imagePath": real_image.name,
            "imageHeight": 100,
            "imageWidth": 100,
            "shapes": [{
                "label": "qrcode",
                "points": [[10, 10], [30, 30]],
                "shape_type": "rectangle",
            }],
        }), encoding="utf-8")

        synthetic_image = processed_root / split_name / f"{split_name}_synthetic.jpg"
        synthetic_image.write_bytes(b"synthetic")
        synthetic_json = processed_root / split_name / f"{split_name}_synthetic.json"
        synthetic_json.write_text(json.dumps({
            "imagePath": synthetic_image.name,
            "imageHeight": 100,
            "imageWidth": 100,
            "shapes": [{
                "label": "qrcode",
                "points": [[40, 40], [60, 60]],
                "shape_type": "rectangle",
            }],
        }), encoding="utf-8")

    output_dir = tmp_path / "dataset"
    summary = build_dataset_from_splits(splits_dir=splits_root, processed_dir=processed_root, output_dir=output_dir)

    assert summary["splits"]["train"]["total_images"] == 3
    assert (output_dir / "dataset.yaml").exists()
    assert (output_dir / "images" / "train" / "train_negative.jpg").exists()
    assert (output_dir / "labels" / "train" / "train_negative.txt").read_text(encoding="utf-8") == ""

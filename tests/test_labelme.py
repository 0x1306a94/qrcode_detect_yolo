import json
from pathlib import Path

from qrcode_detector.labelme import export_labelme_directory_to_yolo


def test_export_labelme_directory_to_yolo_builds_dataset(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for index in range(3):
        image_path = input_dir / f"sample_{index}.jpg"
        image_path.write_bytes(b"fake-image")
        json_path = input_dir / f"sample_{index}.json"
        payload = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [
                {
                    "label": "qrcode",
                    "points": [[10, 20], [110, 120]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            ],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": 400,
            "imageWidth": 300,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    output_dir = tmp_path / "dataset"
    counts = export_labelme_directory_to_yolo(input_dir=input_dir, output_dir=output_dir)

    assert counts["total"] == 3
    assert (output_dir / "dataset.yaml").exists()
    assert list((output_dir / "images" / "train").glob("*.jpg"))
    assert list((output_dir / "labels" / "train").glob("*.txt"))

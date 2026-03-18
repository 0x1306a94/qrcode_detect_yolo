from pathlib import Path

from qrcode_detector.dataset import ManifestRecord, split_records


def test_split_records_keeps_validation_and_test_when_possible() -> None:
    records = [
        ManifestRecord(image_path=Path(f"/tmp/{index}.jpg"), boxes=(), is_negative=True)
        for index in range(10)
    ]

    splits = split_records(records, train_ratio=0.8, validation_ratio=0.1, seed=1)

    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1

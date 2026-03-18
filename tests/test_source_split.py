from pathlib import Path

from qrcode_detector.source_split import SourceSplitConfig, split_training_sources


def test_split_training_sources_creates_expected_layout(tmp_path: Path) -> None:
    origins_dir = tmp_path / "origins"
    reals_dir = tmp_path / "reals"
    origins_dir.mkdir()
    reals_dir.mkdir()

    for index in range(10):
        (origins_dir / f"origin_{index}.jpg").write_bytes(b"origin")
    for index in range(4):
        (reals_dir / f"real_{index}.jpg").write_bytes(b"real")

    summary = split_training_sources(
        origins_dir=origins_dir,
        reals_dir=reals_dir,
        output_dir=tmp_path / "splits",
        config=SourceSplitConfig(
            train_ratio=0.8,
            validation_ratio=0.1,
            synthesis_source_ratio=0.5,
            seed=1,
        ),
    )

    assert summary["splits"]["train"]["negative_origins"] == 4
    assert summary["splits"]["train"]["synthetic_sources"] == 4
    assert summary["splits"]["train"]["real_positives"] == 2
    assert (tmp_path / "splits" / "split_summary.json").exists()

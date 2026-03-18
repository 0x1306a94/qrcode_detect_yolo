from pathlib import Path

from qrcode_detector.download import _build_file_name, _load_urls


def test_load_urls_skips_duplicates_and_blank_lines(tmp_path: Path) -> None:
    csv_path = tmp_path / "images.csv"
    csv_path.write_text(
        "https://example.com/a.jpg\n\nhttps://example.com/a.jpg\nhttps://example.com/b.png\n",
        encoding="utf-8",
    )

    urls = _load_urls(csv_path)

    assert urls == [
        "https://example.com/a.jpg",
        "https://example.com/b.png",
    ]


def test_build_file_name_adds_hash_suffix() -> None:
    file_name = _build_file_name("https://example.com/path/test-image.jpg", ".jpg")
    assert file_name.startswith("test-image_")
    assert file_name.endswith(".jpg")

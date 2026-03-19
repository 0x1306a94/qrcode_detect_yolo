from pathlib import Path

from qrcode_detector.synthetic import (
    SyntheticPlacement,
    SyntheticQRCodeConfig,
    _random_qrcode_content,
    _resolve_placement_bounds,
    write_labelme_rectangle,
)


def test_write_labelme_rectangle_outputs_expected_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "sample.json"
    placement = SyntheticPlacement(
        image_width=1000,
        image_height=800,
        x1=100,
        y1=600,
        x2=220,
        y2=720,
    )

    write_labelme_rectangle(output_path, tmp_path / "sample.jpg", placement)

    content = output_path.read_text(encoding="utf-8")
    assert '"label": "qrcode"' in content
    assert '"shape_type": "rectangle"' in content
    assert '"imageHeight": 800' in content
    assert '"imageWidth": 1000' in content


def test_synthetic_qrcode_config_validate_accepts_default() -> None:
    config = SyntheticQRCodeConfig()
    config.validate()


def test_random_qrcode_content_respects_requested_length_range() -> None:
    import random

    content = _random_qrcode_content(random.Random(1), (40, 40))

    assert len(content) == 40


def test_resolve_placement_bounds_supports_center_region() -> None:
    x_start, x_end, y_start, y_end = _resolve_placement_bounds(
        placement_region="center",
        image_width=1000,
        image_height=800,
        target_size=120,
        corner_ratio=0.35,
        center_region_x_range=(0.25, 0.75),
        center_region_y_range=(0.35, 0.75),
    )

    assert (x_start, x_end, y_start, y_end) == (250, 750, 280, 600)

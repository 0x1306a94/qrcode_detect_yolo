from pathlib import Path

from qrcode_detector.synthetic import SyntheticPlacement, write_labelme_rectangle


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

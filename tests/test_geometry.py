from qrcode_detector.geometry import apply_nms, box_iou, is_bottom_corner_box
from qrcode_detector.types import BoundingBox


def test_box_iou_returns_expected_overlap() -> None:
    first = BoundingBox(0, 0, 100, 100, 0.9)
    second = BoundingBox(50, 50, 150, 150, 0.8)
    assert round(box_iou(first, second), 4) == 0.1429


def test_apply_nms_keeps_best_overlapping_box() -> None:
    boxes = (
        BoundingBox(0, 0, 100, 100, 0.9),
        BoundingBox(10, 10, 110, 110, 0.8),
        BoundingBox(200, 200, 260, 260, 0.7),
    )
    selected = apply_nms(boxes, iou_threshold=0.5, max_boxes=10)
    assert len(selected) == 2
    assert selected[0].score == 0.9
    assert selected[1].score == 0.7


def test_is_bottom_corner_box_detects_bottom_left_region() -> None:
    box = BoundingBox(10, 650, 110, 750, 0.6)
    assert is_bottom_corner_box(box, image_width=1000, image_height=800, corner_ratio=0.35)

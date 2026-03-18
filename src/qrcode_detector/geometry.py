from __future__ import annotations

from typing import Iterable

from .types import BoundingBox


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    overlap_x1 = max(first.x1, second.x1)
    overlap_y1 = max(first.y1, second.y1)
    overlap_x2 = min(first.x2, second.x2)
    overlap_y2 = min(first.y2, second.y2)

    overlap_width = max(0.0, overlap_x2 - overlap_x1)
    overlap_height = max(0.0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height
    union_area = first.area + second.area - overlap_area

    if union_area <= 0.0:
        return 0.0
    return overlap_area / union_area


def apply_nms(boxes: Iterable[BoundingBox], iou_threshold: float, max_boxes: int) -> tuple[BoundingBox, ...]:
    selected: list[BoundingBox] = []
    remaining = sorted(boxes, key=lambda item: item.score, reverse=True)

    while remaining and len(selected) < max_boxes:
        candidate = remaining.pop(0)
        selected.append(candidate)
        remaining = [
            box for box in remaining
            if box_iou(candidate, box) < iou_threshold
        ]

    return tuple(selected)


def is_bottom_corner_box(box: BoundingBox, image_width: int, image_height: int, corner_ratio: float) -> bool:
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0

    bottom_threshold = image_height * (1.0 - corner_ratio)
    left_threshold = image_width * corner_ratio
    right_threshold = image_width * (1.0 - corner_ratio)

    return center_y >= bottom_threshold and (center_x <= left_threshold or center_x >= right_threshold)

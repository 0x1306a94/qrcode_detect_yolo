from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DetectionConfig:
    target_size: int = 1024
    confidence_floor: float = 0.15
    business_threshold: float = 0.45
    iou_threshold: float = 0.5
    corner_bonus: float = 0.08
    corner_ratio: float = 0.35
    max_boxes: int = 20

    def validate(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        if not 0.0 <= self.business_threshold <= 1.0:
            raise ValueError("business_threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0.0 <= self.corner_bonus <= 1.0:
            raise ValueError("corner_bonus must be in [0, 1]")
        if not 0.0 < self.corner_ratio <= 1.0:
            raise ValueError("corner_ratio must be in (0, 1]")
        if self.max_boxes <= 0:
            raise ValueError("max_boxes must be positive")

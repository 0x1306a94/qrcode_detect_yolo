from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "qrcode"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(slots=True, frozen=True)
class DetectionResult:
    has_qrcode: bool
    score: float
    elapsed_ms: float
    read_elapsed_ms: float
    predict_elapsed_ms: float
    postprocess_elapsed_ms: float
    boxes: tuple[BoundingBox, ...] = field(default_factory=tuple)

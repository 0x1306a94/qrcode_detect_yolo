from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DetectionConfig
from .geometry import apply_nms, is_bottom_corner_box
from .types import BoundingBox, DetectionResult

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


class QRCodeDetector:
    def __init__(self, model_path: str | Path, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.config.validate()
        self.model_path = str(model_path)
        self._model = self._load_model(self.model_path)

    def _load_model(self, model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Ultralytics is required for inference. Install with: pip install -e '.[inference]'"
            ) from exc
        return YOLO(model_path)

    def detect(self, image_path: str | Path) -> DetectionResult:
        width, height = self._read_image_size(image_path)
        predictions = self._predict(image_path)
        boxes = self._post_process(predictions, width, height)
        top_score = boxes[0].score if boxes else 0.0
        return DetectionResult(
            has_qrcode=top_score >= self.config.business_threshold,
            score=top_score,
            boxes=boxes,
        )

    def _read_image_size(self, image_path: str | Path) -> tuple[int, int]:
        if Image is None:  # pragma: no cover
            raise RuntimeError(
                "Pillow is required to read image metadata. Install with: pip install -e '.[inference]'"
            )
        with Image.open(image_path) as image:
            return image.size

    def _predict(self, image_path: str | Path) -> Any:
        return self._model.predict(
            source=str(image_path),
            imgsz=self.config.target_size,
            conf=self.config.confidence_floor,
            verbose=False,
        )

    def _post_process(self, predictions: Any, image_width: int, image_height: int) -> tuple[BoundingBox, ...]:
        raw_boxes: list[BoundingBox] = []
        for prediction in predictions:
            box_data = getattr(prediction, "boxes", None)
            if box_data is None:
                continue
            coordinates = box_data.xyxy.tolist()
            scores = box_data.conf.tolist()
            for index, coordinate in enumerate(coordinates):
                box = BoundingBox(
                    x1=float(coordinate[0]),
                    y1=float(coordinate[1]),
                    x2=float(coordinate[2]),
                    y2=float(coordinate[3]),
                    score=float(scores[index]),
                )
                if is_bottom_corner_box(box, image_width, image_height, self.config.corner_ratio):
                    box = replace(box, score=min(1.0, box.score + self.config.corner_bonus))
                raw_boxes.append(box)

        filtered_boxes = apply_nms(
            boxes=raw_boxes,
            iou_threshold=self.config.iou_threshold,
            max_boxes=self.config.max_boxes,
        )
        return tuple(sorted(filtered_boxes, key=lambda item: item.score, reverse=True))

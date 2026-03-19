from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import DetectionConfig
from .geometry import apply_nms, is_bottom_corner_box
from .types import BoundingBox, DetectionResult

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


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

    def detect(
        self,
        image_path: str | Path,
        *,
        draw_on_image: bool = False,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> DetectionResult | tuple[DetectionResult, PILImage]:
        total_start_time = time.perf_counter()

        read_start_time = time.perf_counter()
        width, height = self._read_image_size(image_path)
        read_elapsed_ms = (time.perf_counter() - read_start_time) * 1000.0

        predict_start_time = time.perf_counter()
        predictions = self._predict(image_path)
        predict_elapsed_ms = (time.perf_counter() - predict_start_time) * 1000.0

        postprocess_start_time = time.perf_counter()
        boxes = self._post_process(predictions, width, height)
        postprocess_elapsed_ms = (time.perf_counter() - postprocess_start_time) * 1000.0

        top_score = boxes[0].score if boxes else 0.0
        elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
        result = DetectionResult(
            has_qrcode=top_score >= self.config.business_threshold,
            score=top_score,
            elapsed_ms=elapsed_ms,
            read_elapsed_ms=read_elapsed_ms,
            predict_elapsed_ms=predict_elapsed_ms,
            postprocess_elapsed_ms=postprocess_elapsed_ms,
            boxes=boxes,
        )
        need_draw = draw_on_image or show
        if need_draw and boxes:
            annotated = self.draw_detections_on_image(image_path, result)
            if output_path is not None:
                annotated.save(output_path)
            if show:
                annotated.show()
            return result, annotated
        return result

    def draw_detections_on_image(
        self, image_path: str | Path, result: DetectionResult
    ) -> Any:
        """在图片上绘制检测框、标签和置信度分数，返回标注后的 PIL Image。"""
        if not result.boxes:
            if Image is None:
                raise RuntimeError(
                    "Pillow is required. Install with: pip install -e '.[inference]'"
                )
            return Image.open(image_path).convert("RGB")
        return self._draw_boxes_on_image(image_path, result)

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

    def _draw_boxes_on_image(self, image_path: str | Path, result: DetectionResult) -> Any:
        if Image is None or ImageDraw is None:  # pragma: no cover
            raise RuntimeError(
                "Pillow is required for visualization. Install with: pip install -e '.[inference]'"
            )
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        _font_paths = (
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        )
        font = None
        for path in _font_paths:
            try:
                font = ImageFont.truetype(path, size=16)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

        box_color = (0, 255, 0)  # 绿色边框
        text_bg_color = (0, 255, 0)
        text_color = (0, 0, 0)

        for box in result.boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            label = f"{box.label} {box.score:.2f}"
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_height = bbox[3] - bbox[1]
            draw.rectangle(
                [x1, y1 - text_height - 8, x1 + (bbox[2] - bbox[0]) + 8, y1],
                fill=text_bg_color,
                outline=box_color,
            )
            draw.text((x1 + 4, y1 - text_height - 4), label, fill=text_color, font=font)
        return image

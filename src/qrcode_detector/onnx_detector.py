from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import DetectionConfig
from .geometry import apply_nms, is_bottom_corner_box
from .types import BoundingBox, DetectionResult

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = None  # type: ignore[assignment]

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover
    register_heif_opener = None  # type: ignore[assignment]


class ONNXQRCodeDetector:
    def __init__(self, model_path: str | Path, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.config.validate()
        self.model_path = str(model_path)
        self._session = self._load_session(self.model_path)
        self._input_name = self._session.get_inputs()[0].name

    def _load_session(self, model_path: str) -> Any:
        if ort is None:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required for ONNX inference. Install with: pip install -e '.[server]'"
            )
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def detect(self, image_bytes: bytes) -> DetectionResult:
        total_start_time = time.perf_counter()

        read_start_time = time.perf_counter()
        image, image_width, image_height = self._load_image(image_bytes)
        input_tensor, scale_ratio, pad_x, pad_y = self._preprocess(image)
        read_elapsed_ms = (time.perf_counter() - read_start_time) * 1000.0

        predict_start_time = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: input_tensor})
        predict_elapsed_ms = (time.perf_counter() - predict_start_time) * 1000.0

        postprocess_start_time = time.perf_counter()
        boxes = self._post_process(outputs, image_width, image_height, scale_ratio, pad_x, pad_y)
        postprocess_elapsed_ms = (time.perf_counter() - postprocess_start_time) * 1000.0

        top_score = boxes[0].score if boxes else 0.0
        elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
        return DetectionResult(
            has_qrcode=top_score >= self.config.business_threshold,
            score=top_score,
            elapsed_ms=elapsed_ms,
            read_elapsed_ms=read_elapsed_ms,
            predict_elapsed_ms=predict_elapsed_ms,
            postprocess_elapsed_ms=postprocess_elapsed_ms,
            boxes=boxes,
        )

    def _load_image(self, image_bytes: bytes) -> tuple[Any, int, int]:
        if Image is None:  # pragma: no cover
            raise RuntimeError(
                "Pillow is required to decode images. Install with: pip install -e '.[server]'"
            )
        # Internal service input is trusted. Disable PIL decompression bomb protection
        # so large source images can still be decoded before we resize them for inference.
        Image.MAX_IMAGE_PIXELS = None
        if register_heif_opener is not None:
            register_heif_opener()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
                image_width, image_height = rgb_image.size
                return rgb_image, image_width, image_height
        except UnidentifiedImageError as exc:
            raise ValueError(
                "Unsupported or unreadable image format. For HEIC/HEIF input, install pillow-heif."
            ) from exc

    def _preprocess(self, image: Any) -> tuple[np.ndarray, float, float, float]:
        target_size = self.config.target_size
        image_array = np.asarray(image, dtype=np.uint8)
        image_height, image_width = image_array.shape[:2]

        scale_ratio = min(target_size / image_width, target_size / image_height)
        resized_width = max(1, int(round(image_width * scale_ratio)))
        resized_height = max(1, int(round(image_height * scale_ratio)))

        resized_image = image.resize((resized_width, resized_height))
        resized_array = np.asarray(resized_image, dtype=np.uint8)

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - resized_width) / 2.0
        pad_y = (target_size - resized_height) / 2.0
        left = int(round(pad_x))
        top = int(round(pad_y))
        canvas[top:top + resized_height, left:left + resized_width] = resized_array

        input_tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, scale_ratio, pad_x, pad_y

    def _post_process(
        self,
        outputs: list[np.ndarray],
        image_width: int,
        image_height: int,
        scale_ratio: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[BoundingBox, ...]:
        predictions = self._normalize_predictions(outputs[0])
        raw_boxes: list[BoundingBox] = []

        for prediction in predictions:
            if prediction.shape[0] < 5:
                continue
            score = float(np.max(prediction[4:]))
            if score < self.config.confidence_floor:
                continue

            center_x, center_y, width, height = map(float, prediction[:4])
            x1 = (center_x - width / 2.0 - pad_x) / scale_ratio
            y1 = (center_y - height / 2.0 - pad_y) / scale_ratio
            x2 = (center_x + width / 2.0 - pad_x) / scale_ratio
            y2 = (center_y + height / 2.0 - pad_y) / scale_ratio

            box = BoundingBox(
                x1=max(0.0, min(x1, image_width)),
                y1=max(0.0, min(y1, image_height)),
                x2=max(0.0, min(x2, image_width)),
                y2=max(0.0, min(y2, image_height)),
                score=score,
            )
            if box.width <= 0.0 or box.height <= 0.0:
                continue
            if is_bottom_corner_box(box, image_width, image_height, self.config.corner_ratio):
                box = replace(box, score=min(1.0, box.score + self.config.corner_bonus))
            raw_boxes.append(box)

        filtered_boxes = apply_nms(
            boxes=raw_boxes,
            iou_threshold=self.config.iou_threshold,
            max_boxes=self.config.max_boxes,
        )
        return tuple(sorted(filtered_boxes, key=lambda item: item.score, reverse=True))

    def _normalize_predictions(self, output: np.ndarray) -> np.ndarray:
        prediction = np.asarray(output)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {prediction.shape}")
        if 5 <= prediction.shape[0] <= 10 and prediction.shape[0] != prediction.shape[1]:
            prediction = prediction.transpose(1, 0)
        if prediction.shape[1] < 5:
            raise ValueError(f"Unsupported ONNX prediction layout: {prediction.shape}")
        return prediction

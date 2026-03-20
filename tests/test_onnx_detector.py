import numpy as np

from qrcode_detector.config import DetectionConfig
from qrcode_detector.onnx_detector import ONNXQRCodeDetector


def test_normalize_predictions_accepts_channel_first_output() -> None:
    detector = ONNXQRCodeDetector.__new__(ONNXQRCodeDetector)
    detector.config = DetectionConfig()

    output = np.array([[[10.0, 20.0], [10.0, 20.0], [4.0, 8.0], [4.0, 8.0], [0.9, 0.1]]], dtype=np.float32)
    predictions = detector._normalize_predictions(output)

    assert predictions.shape == (2, 5)
    assert round(float(predictions[0, 4]), 4) == 0.9


def test_normalize_predictions_keeps_row_major_output() -> None:
    detector = ONNXQRCodeDetector.__new__(ONNXQRCodeDetector)
    detector.config = DetectionConfig()

    output = np.array(
        [[10.0, 10.0, 4.0, 4.0, 0.9], [20.0, 20.0, 8.0, 8.0, 0.1]],
        dtype=np.float32,
    )
    predictions = detector._normalize_predictions(output)

    assert predictions.shape == (2, 5)
    assert round(float(predictions[0, 4]), 4) == 0.9


def test_post_process_decodes_single_class_boxes() -> None:
    detector = ONNXQRCodeDetector.__new__(ONNXQRCodeDetector)
    detector.config = DetectionConfig(confidence_floor=0.2, corner_bonus=0.0)

    outputs = [np.array([[[50.0], [60.0], [20.0], [30.0], [0.95]]], dtype=np.float32)]
    boxes = detector._post_process(
        outputs=outputs,
        image_width=100,
        image_height=100,
        scale_ratio=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )

    assert len(boxes) == 1
    assert round(boxes[0].x1, 2) == 40.0
    assert round(boxes[0].y1, 2) == 45.0
    assert round(boxes[0].x2, 2) == 60.0
    assert round(boxes[0].y2, 2) == 75.0

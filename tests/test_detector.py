from qrcode_detector.config import DetectionConfig
from qrcode_detector.detector import QRCodeDetector
from qrcode_detector.types import BoundingBox


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    def __init__(self, coordinates, scores):
        self.xyxy = _FakeTensor(coordinates)
        self.conf = _FakeTensor(scores)


class _FakePrediction:
    def __init__(self, coordinates, scores):
        self.boxes = _FakeBoxes(coordinates, scores)


def test_post_process_applies_corner_bonus() -> None:
    detector = QRCodeDetector.__new__(QRCodeDetector)
    detector.config = DetectionConfig(corner_bonus=0.1, business_threshold=0.5)
    predictions = [
        _FakePrediction(
            coordinates=[[10, 650, 110, 750], [400, 50, 500, 150]],
            scores=[0.45, 0.44],
        )
    ]

    boxes = detector._post_process(predictions, image_width=1000, image_height=800)

    assert len(boxes) == 2
    assert boxes[0] == BoundingBox(10.0, 650.0, 110.0, 750.0, 0.55)
    assert boxes[1] == BoundingBox(400.0, 50.0, 500.0, 150.0, 0.44)


def test_detect_returns_elapsed_ms() -> None:
    detector = QRCodeDetector.__new__(QRCodeDetector)
    detector.config = DetectionConfig(corner_bonus=0.1, business_threshold=0.5)
    detector._read_image_size = lambda _: (1000, 800)
    detector._predict = lambda _: [
        _FakePrediction(
            coordinates=[[10, 650, 110, 750]],
            scores=[0.45],
        )
    ]

    result = detector.detect("fake.jpg")

    assert result.has_qrcode is True
    assert result.score == 0.55
    assert result.elapsed_ms >= 0.0
    assert result.read_elapsed_ms >= 0.0
    assert result.predict_elapsed_ms >= 0.0
    assert result.postprocess_elapsed_ms >= 0.0

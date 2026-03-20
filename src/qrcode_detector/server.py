from dataclasses import asdict
from typing import Any
import time

from .config import DetectionConfig
from .onnx_detector import ONNXQRCodeDetector


def create_app(model_path: str, config: DetectionConfig | None = None) -> Any:
    try:
        import base64
        from urllib.error import HTTPError, URLError
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        from fastapi import FastAPI, File, HTTPException, Request as FastAPIRequest, UploadFile
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required for serving. Install with: pip install -e '.[server]'"
        ) from exc

    detector = ONNXQRCodeDetector(model_path=model_path, config=config)
    app = FastAPI(title="QRCode Detect YOLO", version="0.1.0")
    download_timeout_seconds = 10
    max_download_bytes = 10 * 1024 * 1024

    def build_response(
        result: Any,
        input_type: str,
        image_url_download_elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        return {
            "input_type": input_type,
            "image_url_download_elapsed_ms": image_url_download_elapsed_ms,
            "has_qrcode": result.has_qrcode,
            "score": result.score,
            "elapsed_ms": result.elapsed_ms,
            "read_elapsed_ms": result.read_elapsed_ms,
            "predict_elapsed_ms": result.predict_elapsed_ms,
            "postprocess_elapsed_ms": result.postprocess_elapsed_ms,
            "boxes": [asdict(box) for box in result.boxes],
        }

    def fetch_image_bytes(image_url: str) -> bytes:
        parsed_url = urlparse(image_url)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("image_url only supports http and https")
        request = Request(
            image_url,
            headers={"User-Agent": "qrcode-detect-yolo/0.1.0"},
        )
        try:
            with urlopen(request, timeout=download_timeout_seconds) as response:
                data = response.read(max_download_bytes + 1)
        except (HTTPError, URLError) as exc:
            raise ValueError(f"Failed to download image_url: {exc}") from exc
        if len(data) > max_download_bytes:
            raise ValueError("Downloaded image exceeds 10MB limit")
        return data

    def decode_base64_image(image_base64: str) -> bytes:
        payload = image_base64.strip()
        if "," in payload and payload.lower().startswith("data:"):
            payload = payload.split(",", 1)[1]
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError("image_base64 is not valid base64") from exc

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/detect")
    async def detect(
        request: FastAPIRequest,
        file: UploadFile | None = File(default=None),
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        image_bytes: bytes | None = None
        input_type: str | None = None
        image_url_download_elapsed_ms: float | None = None
        provided_count = int(file is not None)
        if payload is not None:
            provided_count += int(bool(payload.get("image_url")))
            provided_count += int(bool(payload.get("image_base64")))
        if provided_count > 1:
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one input: file or image_url or image_base64",
            )

        if file is not None:
            input_type = "file"
            image_bytes = await file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
        elif payload is not None and payload.get("image_url"):
            input_type = "image_url"
            try:
                download_start_time = time.perf_counter()
                image_bytes = fetch_image_bytes(str(payload["image_url"]))
                image_url_download_elapsed_ms = (time.perf_counter() - download_start_time) * 1000.0
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif payload is not None and payload.get("image_base64"):
            input_type = "image_base64"
            try:
                image_bytes = decode_base64_image(str(payload["image_base64"]))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Missing input image. Provide multipart file upload or JSON body with image_url/image_base64. content-type={content_type}",
            )

        try:
            result = detector.detect(image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return build_response(
            result,
            input_type=input_type or "unknown",
            image_url_download_elapsed_ms=image_url_download_elapsed_ms,
        )

    return app

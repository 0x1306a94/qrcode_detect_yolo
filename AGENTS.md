# Repository Guidelines

## Project Structure & Module Organization

Source code lives in `src/qrcode_detector/`. Key modules include `cli.py` for command entrypoints, `synthetic.py` for QR synthesis, `dataset_build.py` and `labelme.py` for dataset assembly, and `detector.py` for inference. Tests are in `tests/` and follow the module split (`test_detector.py`, `test_synthetic.py`, etc.). Generated assets should stay outside source code, typically under `auto_generated_test_images/`, `dataset/`, `runs/`, or `train_out/`.

## Build, Test, and Development Commands

- `python3 -m pip install -e .`
  Installs the package in editable mode.
- `python3 -m pip install -e ".[synthetic,inference,dev]"`
  Installs QR generation, inference, and test dependencies.
- `python3 src/qrcode_detector/cli.py --help`
  Lists all supported local workflows.
- `python3 -m compileall src tests`
  Fast syntax check for all modules.
- `pytest -q`
  Runs the test suite.
- `yolo detect train data=./dataset/dataset.yaml model=./train_out/yolov8n.pt imgsz=1024 epochs=20 batch=4 device=cpu`
  Starts a local training run.

## Coding Style & Naming Conventions

Use Python 3.11+ with 4-space indentation. Prefer short, explicit names such as `output_dir`, `image_path`, and `business_threshold`; avoid unclear abbreviations. Match the existing style: dataclasses for structured results, small pure helper functions, and English code/comments. Use `apply_patch` for manual edits. Prefer extending existing modules over creating parallel duplicate flows.

## Testing Guidelines

The project uses `pytest`. Add tests for every behavior change, especially CLI parsing, dataset conversion, and synthesis rules. Name tests `test_<behavior>.py` and keep assertions concrete. For quick validation during development, `python3 -m compileall src tests` is the minimum gate; `pytest -q` is the expected check before submitting changes.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit messages such as `优化合成`, `检测添加可视化以及耗时统计`, and `init 02`. Keep messages concise and focused on one change. For pull requests, include: purpose, affected commands or paths, how you tested it, and sample output when changing dataset generation, training, or visualization behavior.

## Data & Artifact Notes

Do not commit large generated datasets, model weights, or temporary outputs unless explicitly needed. Treat `images.csv`, downloaded images, and trained weights as reproducible artifacts. Keep source inputs (`origins`, `reals`) separate from generated outputs (`processed`, `dataset`, `runs`).

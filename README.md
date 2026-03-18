# 二维码检测 YOLO 工程

这个仓库提供了一个最小可用的 Python 工程骨架，用于管理平台上传图片中的二维码检测训练数据准备和线上推理接入。

## 功能范围

- 单类别检测：`qrcode`
- 导出 YOLO 可训练数据集
- 生成少量合成二维码贴图样本作为增强
- 推理阶段支持左下 / 右下角区域分数加权

## 安装

安装基础包：

```bash
python3 -m pip install -e .
```

安装推理依赖：

```bash
python3 -m pip install -e ".[inference]"
```

安装合成数据依赖：

```bash
python3 -m pip install -e ".[synthetic]"
```

安装开发依赖：

```bash
python3 -m pip install -e ".[dev]"
```

## Manifest 格式

数据集切分和导出命令依赖 JSONL manifest 文件。每一行示例如下：

```json
{"image_path": "/abs/path/to/image.jpg", "boxes": [{"x1": 120, "y1": 400, "x2": 280, "y2": 560}], "is_negative": false}
```

对于不包含二维码的图片：

```json
{"image_path": "/abs/path/to/no_qr.jpg", "boxes": [], "is_negative": true}
```

## 命令

根据 manifest 导出 YOLO 数据集：

```bash
qrcode-detect export-dataset \
  --manifest /abs/path/to/manifest.jsonl \
  --output-dir /abs/path/to/output_dataset
```

把一张二维码图贴到海报图上，生成一张合成样本：

```bash
qrcode-detect synthesize \
  --background /abs/path/to/background.jpg \
  --qrcode /abs/path/to/qrcode.png \
  --output /abs/path/to/output.jpg
```

使用 Ultralytics YOLO 权重执行检测：

```bash
qrcode-detect detect \
  --model /abs/path/to/best.pt \
  --image /abs/path/to/input.jpg
```

## 建议训练流程

1. 导出历史上传图片。
2. 对真实正样本按单类别 `qrcode` 进行标注。
3. 从真实海报图中补充困难负样本。
4. 只生成少量合成样本作为增强，不把它当主数据集。
5. 导出 YOLO 数据集。
6. 训练轻量 YOLO 模型，并按高召回目标校准阈值。

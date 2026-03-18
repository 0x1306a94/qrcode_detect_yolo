# 二维码检测 YOLO 工程

这个仓库提供了一个最小可用的 Python 工程骨架，用于管理平台上传图片中的二维码检测训练数据准备和线上推理接入。

如果你要批量制造合成样本，当前支持的方式是：

- 指定一个原图目录
- 遍历每张原图
- 为每张图随机生成一个二维码
- 将二维码绘制到左下或右下区域
- 保存成新图片
- 可选自动生成 Labelme 矩形标注 JSON

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

把 `Labelme JSON + 图片` 目录直接转换成 YOLO 数据集：

```bash
python3 src/qrcode_detector/cli.py export-labelme-dataset \
  --input-dir ./auto_generated_test_images/synthesizes \
  --output-dir ./dataset_demo
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

批量遍历目录，为每张图生成一个合成二维码样本，并写出 Labelme 标注：

```bash
qrcode-detect synthesize-directory \
  --input-dir /abs/path/to/source_images \
  --output-dir /abs/path/to/synthetic_images \
  --recursive \
  --write-labelme-json
```

输出目录中会包含：

- 合成后的图片文件
- 对应的 Labelme `*.json`
- 程序生成的二维码原图，位于 `_generated_qrcodes/`

说明：

- 如果二维码是程序贴上去的，原则上不需要再人工框选，因为矩形框已知，自动生成标注更准确。
- 如果你想人工复核，直接用 Labelme 打开自动生成的 JSON 修改即可。

## 最小跑通流程

1. 生成合成样本和 Labelme JSON：

```bash
python3 src/qrcode_detector/cli.py synthesize-directory \
  --input-dir ./auto_generated_test_images/origins \
  --output-dir ./auto_generated_test_images/synthesizes \
  --write-labelme-json
```

1. 转成 YOLO 数据集：

```bash
python3 src/qrcode_detector/cli.py export-labelme-dataset \
  --input-dir ./auto_generated_test_images/synthesizes \
  --output-dir ./dataset_demo
```

1. 用 Ultralytics 开始训练：

```bash
yolo detect train \
  data=./dataset_demo/dataset.yaml \
  model=train_out/yolov8n.pt \
  imgsz=640 \
  epochs=20 \
  batch=4 \
  device=cpu

# Apple Silicon MPS
device=mps
```

说明：

- 这条链路适合先跑通工程，不代表最终训练数据质量足够上线。

## 下载原图

如果 `images.csv` 每行都是一个图片链接，可以直接下载到 `auto_generated_test_images/origins`：

```bash
python3 src/qrcode_detector/cli.py download-images \
  --csv ./images.csv
```

也可以显式指定输出目录：

```bash
python3 src/qrcode_detector/cli.py download-images \
  --csv ./images.csv \
  --output-dir ./auto_generated_test_images/origins
```

脚本行为：

- 自动跳过空行和重复 URL
- 按稳定文件名保存，避免同名覆盖
- 如果目标文件已存在，会跳过
- 最后输出下载统计

## 切分训练数据源

如果你已经准备好了：

- `auto_generated_test_images/origins`：真实无码图
- `auto_generated_test_images/reals`：真实有码图

可以先把它们切分成训练来源目录：

```bash
python3 src/qrcode_detector/cli.py split-training-sources
```

默认会输出到 `auto_generated_test_images/splits`，结构如下：

```text
auto_generated_test_images/splits/
  train/
    negative_origins/
    synthetic_sources/
    real_positives/
  val/
    negative_origins/
    synthetic_sources/
    real_positives/
  test/
    negative_origins/
    synthetic_sources/
    real_positives/
```

说明：

- `negative_origins/`：直接作为无码负样本
- `synthetic_sources/`：后续用于生成合成有码图
- `real_positives/`：真实有码图，建议人工标注
- 详细统计会写到 `split_summary.json`

## 从 splits 生成最终数据集

如果你已经有：

- `auto_generated_test_images/processed/{train,val,test}`：合成正样本和 Labelme JSON
- `auto_generated_test_images/splits/{train,val,test}/real_positives`：真实正样本和 Labelme JSON
- `auto_generated_test_images/splits/{train,val,test}/negative_origins`：真实负样本

可以直接生成最终 YOLO 数据集：

```bash
python3 src/qrcode_detector/cli.py build-dataset-from-splits
```

默认输出到 `./dataset`，结构为：

```text
dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  dataset.yaml
```

## 建议训练流程

1. 导出历史上传图片。
2. 对真实正样本按单类别 `qrcode` 进行标注。
3. 从真实海报图中补充困难负样本。
4. 只生成少量合成样本作为增强，不把它当主数据集。
5. 导出 YOLO 数据集。
6. 训练轻量 YOLO 模型，并按高召回目标校准阈值。


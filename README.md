# Circle Detector Training with YOLO

This repository contains a minimal pipeline to train a circle detector using **YOLOv8**.

## What it does

- Generates a synthetic dataset of images with circles.
- Writes YOLO-format labels (`class x_center y_center width height`).
- Creates the dataset YAML config expected by Ultralytics YOLO.
- Trains a detector using a pretrained YOLOv8 model as a starting point.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_yolo_circle_detector.py --output-dir runs/circle --epochs 50
```

## Notes

- The model is trained for one class: `circle`.
- Synthetic data is useful for bootstrapping, but for production quality you should include real images.

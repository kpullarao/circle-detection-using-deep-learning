"""Train a circle detector using Ultralytics YOLOv8.

The script can generate a synthetic circle dataset and train a one-class YOLO model.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO circle detector")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/circle"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _draw_random_background(image_size: int) -> np.ndarray:
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

    for _ in range(random.randint(3, 12)):
        x1, y1 = random.randint(0, image_size - 1), random.randint(0, image_size - 1)
        x2, y2 = random.randint(0, image_size - 1), random.randint(0, image_size - 1)
        color = tuple(random.randint(180, 245) for _ in range(3))
        thickness = random.randint(1, 3)
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    noise = np.random.normal(loc=0, scale=8, size=image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _generate_example(image_size: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    image = _draw_random_background(image_size)

    radius = random.randint(image_size // 16, image_size // 6)
    cx = random.randint(radius + 1, image_size - radius - 2)
    cy = random.randint(radius + 1, image_size - radius - 2)

    circle_color = tuple(random.randint(20, 170) for _ in range(3))
    fill_or_outline = random.choice([-1, random.randint(2, 6)])
    cv2.circle(image, (cx, cy), radius, circle_color, fill_or_outline)

    x_center = cx / image_size
    y_center = cy / image_size
    width = (2 * radius) / image_size
    height = (2 * radius) / image_size
    return image, (x_center, y_center, width, height)


def _write_split(base_dir: Path, split: str, samples: int, image_size: int) -> None:
    images_dir = base_dir / "images" / split
    labels_dir = base_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(samples):
        image, bbox = _generate_example(image_size=image_size)
        image_name = f"{split}_{idx:05d}.jpg"
        label_name = f"{split}_{idx:05d}.txt"

        cv2.imwrite(str(images_dir / image_name), image)

        x_center, y_center, width, height = bbox
        with open(labels_dir / label_name, "w", encoding="utf-8") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def _write_dataset_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "circle"},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return yaml_path


def generate_dataset(dataset_dir: Path, image_size: int, train_samples: int, val_samples: int) -> Path:
    _write_split(dataset_dir, split="train", samples=train_samples, image_size=image_size)
    _write_split(dataset_dir, split="val", samples=val_samples, image_size=image_size)
    return _write_dataset_yaml(dataset_dir)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = generate_dataset(
        dataset_dir=dataset_dir,
        image_size=args.image_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        project=str(output_dir),
        name="train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

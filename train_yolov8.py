#!/usr/bin/env python3
"""
Train YOLOv8 model on the prepared dataset.

The script expects:
  - Images arranged under prepared_dataset/<class_slug>/<image>
  - YOLO label files under prepared_dataset/labels_yolo/<class_slug>/<image>.txt
  - prepared_dataset/yolo_classes.txt describing class ids/names
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
    _YOLO_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET_DIR = Path("prepared_dataset")
DEFAULT_OUTPUT_DIR = Path("runs")

logger = logging.getLogger("yolov8-trainer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 detector using the prepared dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory that contains the downloaded images (default: prepared_dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where checkpoints and logs should be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model size (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of samples used for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument(
        "--device",
        default="",
        help="Device to run on (e.g. 0, 0,1, cpu). Empty string defaults to CUDA when available.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if _YOLO_IMPORT_ERROR is not None:
        logger.error(
            "Ultralytics is required to train YOLOv8 (%s). Install it via 'pip install ultralytics'.",
            _YOLO_IMPORT_ERROR,
        )
        return 1

    if not args.dataset_dir.exists():
        logger.error("Dataset directory %s does not exist", args.dataset_dir)
        return 1

    classes_file = args.dataset_dir / "yolo_classes.txt"
    if not classes_file.exists():
        logger.error("Classes file %s does not exist", classes_file)
        return 1

    # Load class mapping
    class_map = load_class_mapping(classes_file)
    if not class_map:
        logger.error("No classes found in %s", classes_file)
        return 1

    # Create YOLO dataset structure
    logger.info("Preparing YOLOv8 dataset structure...")
    yolo_dir = Path("yolo_dataset")
    yolo_dir.mkdir(exist_ok=True)

    train_images_dir = yolo_dir / "images" / "train"
    val_images_dir = yolo_dir / "images" / "val"
    train_labels_dir = yolo_dir / "labels" / "train"
    val_labels_dir = yolo_dir / "labels" / "val"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image-label pairs
    samples = discover_samples(args.dataset_dir, args.dataset_dir / "labels_yolo")
    if not samples:
        logger.error("No matching image/label pairs found. Did you run prepare_dataset.py with --bbox-format yolo?")
        return 1

    logger.info("Found %d samples", len(samples))

    # Split dataset
    random.seed(args.seed)
    random.shuffle(samples)
    split_index = int(len(samples) * args.train_split)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    logger.info("Split: %d train / %d val", len(train_samples), len(val_samples))

    # Copy files to YOLO structure
    for sample in train_samples:
        shutil.copy2(sample["image"], train_images_dir / sample["image"].name)
        shutil.copy2(sample["label"], train_labels_dir / sample["label"].name)

    for sample in val_samples:
        shutil.copy2(sample["image"], val_images_dir / sample["image"].name)
        shutil.copy2(sample["label"], val_labels_dir / sample["label"].name)

    # Create data.yaml
    data_yaml_path = yolo_dir / "data.yaml"
    create_data_yaml(data_yaml_path, class_map, yolo_dir)

    # Initialize and train model
    logger.info("Loading YOLOv8 model: %s", args.model)
    model = YOLO(args.model)

    logger.info("Starting training...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=str(args.output_dir),
        name="yolov8_detection",
        patience=args.patience,
        save=True,
        save_period=10,
        verbose=True,
    )

    logger.info("Training complete!")
    logger.info("Best model saved to: %s", args.output_dir / "yolov8_detection" / "weights" / "best.pt")

    return 0


def load_class_mapping(path: Path) -> dict[int, str]:
    """Load class mapping from CSV file."""
    mapping: dict[int, str] = {}
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                class_id_str, label = line.split(",", 1)
                mapping[int(class_id_str.strip())] = label.strip()
            except ValueError:
                logger.warning("Skipping malformed class line: %s", line)
    return mapping


def discover_samples(images_dir: Path, labels_dir: Path) -> list[dict]:
    """Find all image-label pairs."""
    samples = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.parent / f"{image_path.stem}.txt"
        if label_path.exists() and has_valid_labels(label_path):
            samples.append({"image": image_path, "label": label_path})
    return samples


def has_valid_labels(label_path: Path) -> bool:
    """Check if label file has valid YOLO format labels."""
    try:
        with label_path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if len(line.split()) == 5:
                    return True
    except OSError:
        return False
    return False


def create_data_yaml(path: Path, class_map: dict[int, str], base_dir: Path) -> None:
    """Create data.yaml file for YOLOv8."""
    with path.open("w") as f:
        f.write(f"path: {base_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write(f"nc: {len(class_map)}\n")
        f.write(f"names: {list(class_map.values())}\n")
    logger.info("Created data.yaml at %s", path)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run YOLOv8 inference on random images from the dataset.

The script expects:
  - A trained YOLOv8 model checkpoint (*.pt)
  - Images directory (e.g., validation set)
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

try:
    from ultralytics import YOLO
    _YOLO_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc

try:
    from PIL import Image
    _PIL_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = Path("yolo_dataset/images/val")
DEFAULT_MODEL_PATH = Path("runs/yolov8_detection/weights/best.pt")
DEFAULT_OUTPUT_DIR = Path("predictions")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_NUM_IMAGES = 10

logger = logging.getLogger("yolov8-predictor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on random images from the dataset."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing images (default: yolo_dataset/images/val).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLOv8 model checkpoint (*.pt file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save prediction results.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of random images to predict on (default: 10).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence threshold for displaying predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image selection.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if _YOLO_IMPORT_ERROR is not None:
        logger.error(
            "Ultralytics is required to run YOLOv8 inference (%s). Install it via 'pip install ultralytics'.",
            _YOLO_IMPORT_ERROR,
        )
        return 1

    if _PIL_IMPORT_ERROR is not None:
        logger.error(
            "Pillow is required to read images (%s). Install it via 'pip install pillow'.",
            _PIL_IMPORT_ERROR,
        )
        return 1

    if not args.images_dir.exists():
        logger.error("Images directory %s does not exist", args.images_dir)
        return 1
    if not args.model_path.exists():
        logger.error("Model checkpoint %s does not exist", args.model_path)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # Find all images
    images = discover_images(args.images_dir)
    if not images:
        logger.error("No images found in %s", args.images_dir)
        return 1

    logger.info("Found %d images in %s", len(images), args.images_dir)

    # Select random images
    num_to_select = min(args.num_images, len(images))
    selected_images = random.sample(images, num_to_select)
    logger.info("Selected %d random images for prediction", num_to_select)

    # Load model
    logger.info("Loading YOLOv8 model from %s", args.model_path)
    model = YOLO(str(args.model_path))

    # Run predictions on each selected image
    print("\n" + "=" * 80)
    print(f"Running predictions on {num_to_select} random images")
    print("=" * 80)

    for idx, selected_image in enumerate(selected_images, 1):
        print(f"\n[{idx}/{num_to_select}] Image: {selected_image.name}")
        print("-" * 80)

        # Run inference
        results = model.predict(
            source=str(selected_image),
            conf=args.confidence_threshold,
            save=False,
            verbose=False,
        )

        # Process predictions
        result = results[0]
        boxes = result.boxes

        # Save output with annotations
        output_filename = f"{selected_image.stem}_prediction.jpg"
        output_path = args.output_dir / output_filename

        # Plot results
        annotated_img = result.plot()
        Image.fromarray(annotated_img[..., ::-1]).save(output_path)  # Convert BGR to RGB
        logger.info("Saved prediction to %s", output_path)

        # Print predictions
        if len(boxes) > 0:
            print(f"Found {len(boxes)} detection(s):")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                class_name = model.names[cls_id]
                print(f"  - Class: {class_name:20s} | Confidence: {conf:.4f} | Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        else:
            print("  No detections found.")

    print("\n" + "=" * 80)
    print(f"All predictions saved to: {args.output_dir}")
    print("=" * 80)

    return 0


def discover_images(images_dir: Path) -> list[Path]:
    """Find all images in the directory."""
    images: list[Path] = []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(image_path)
    return images


if __name__ == "__main__":
    raise SystemExit(main())

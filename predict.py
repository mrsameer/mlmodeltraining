#!/usr/bin/env python3
"""
Run inference on a random image from the prepared dataset for a given class.

The script expects:
  - Images arranged under prepared_dataset/<class_slug>/<image>
  - A trained model checkpoint (*.pth) from training_runs/
  - prepared_dataset/yolo_classes.txt describing class ids/names
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - dependency availability checked at runtime
    from PIL import Image, ImageDraw, ImageFont
    _PIL_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc

try:  # pragma: no cover - importing heavy deps is validated at runtime
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import functional as F
    _TORCH_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - handled in main()
    torch = None  # type: ignore[assignment]
    fasterrcnn_resnet50_fpn = None  # type: ignore[assignment]
    FastRCNNPredictor = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET_DIR = Path("prepared_dataset")
DEFAULT_MODEL_PATH = Path("runs/sheath_blight/best_model.pth")
DEFAULT_CLASSES_FILE = Path("prepared_dataset/yolo_classes.txt")
DEFAULT_OUTPUT_DIR = Path("predictions")
DEFAULT_CONFIDENCE_THRESHOLD = 0.1

logger = logging.getLogger("detector-predictor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a random image from a specific class."
    )
    parser.add_argument(
        "--class-name",
        type=str,
        required=True,
        help="Class name (slug) to select an image from (e.g., Sheath_Blight).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory that contains the downloaded images (default: prepared_dataset).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model checkpoint (*.pth file).",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=DEFAULT_CLASSES_FILE,
        help="CSV file with 'class_id,label' pairs (default: prepared_dataset/yolo_classes.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save prediction results.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence threshold for displaying predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. cuda, cuda:1, cpu). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for image selection. If not provided, uses random selection.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if _PIL_IMPORT_ERROR is not None:
        logger.error(
            "Pillow is required to read images (%s). Install it via 'pip install pillow'.",
            _PIL_IMPORT_ERROR,
        )
        return 1

    if _TORCH_IMPORT_ERROR is not None:
        logger.error(
            "Required dependencies missing (%s). Install torch and torchvision to run inference.",
            _TORCH_IMPORT_ERROR,
        )
        return 1

    if not args.dataset_dir.exists():
        logger.error("Dataset directory %s does not exist", args.dataset_dir)
        return 1
    if not args.model_path.exists():
        logger.error("Model checkpoint %s does not exist", args.model_path)
        return 1
    if not args.classes_file.exists():
        logger.error("Classes file %s does not exist", args.classes_file)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    # Load class mapping
    class_map = load_class_mapping(args.classes_file)
    if not class_map:
        logger.error("No classes found in %s", args.classes_file)
        return 1

    # Find images for the specified class
    class_dir = args.dataset_dir / args.class_name
    if not class_dir.exists():
        logger.error("Class directory %s does not exist", class_dir)
        logger.info("Available classes: %s", ", ".join(
            d.name for d in args.dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ))
        return 1

    images = discover_images(class_dir)
    if not images:
        logger.error("No images found in %s", class_dir)
        return 1

    # Select a random image
    selected_image = random.choice(images)
    logger.info("Selected image: %s", selected_image)

    # Load model
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Loading model from %s on device %s", args.model_path, device)

    num_classes = len(class_map) + 1  # background + dataset classes
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Running inference...")

    # Load and preprocess image
    image = Image.open(selected_image).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    # Filter by confidence threshold
    mask = scores >= args.confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    logger.info("Found %d predictions above confidence threshold %.2f", len(boxes), args.confidence_threshold)

    # Draw predictions on image
    output_image = draw_predictions(image, boxes, labels, scores, class_map)

    # Save output
    output_filename = f"{args.class_name}_{selected_image.stem}_prediction.jpg"
    output_path = args.output_dir / output_filename
    output_image.save(output_path)
    logger.info("Saved prediction to %s", output_path)

    # Print predictions
    print("\nPredictions:")
    print("-" * 80)
    for box, label, score in zip(boxes, labels, scores):
        class_name = class_map.get(label - 1, f"unknown_{label}")
        print(f"Class: {class_name:20s} | Confidence: {score:.4f} | Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    print("-" * 80)

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


def discover_images(class_dir: Path) -> list[Path]:
    """Find all images in the class directory."""
    images: list[Path] = []
    for image_path in sorted(class_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(image_path)
    return images


def draw_predictions(
    image: Image.Image,
    boxes: Sequence,
    labels: Sequence,
    scores: Sequence,
    class_map: dict[int, str],
) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=16)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Define colors for different classes
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
    ]

    for box, label, score in zip(boxes, labels, scores):
        class_name = class_map.get(label - 1, f"unknown_{label}")
        color = colors[(label - 1) % len(colors)]

        # Draw bounding box
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline=color,
            width=3,
        )

        # Draw label background
        text = f"{class_name} {score:.2f}"
        bbox = draw.textbbox((box[0], box[1] - 20), text, font=font)
        draw.rectangle(bbox, fill=color)

        # Draw label text
        draw.text(
            (box[0], box[1] - 20),
            text,
            fill="white",
            font=font,
        )

    return image


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Minimal training loop to fine-tune a Faster R-CNN detector on the prepared dataset.

The script expects:
  - Images arranged under prepared_dataset/<class_slug>/<image>
  - YOLO label files under prepared_dataset/labels_yolo/<class_slug>/<image>.txt
  - prepared_dataset/yolo_classes.txt describing class ids/names
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - dependency availability checked at runtime
    from PIL import Image
    _PIL_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc

try:  # pragma: no cover - importing heavy deps is validated at runtime
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import functional as F
    _TORCH_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - handled in main()
    torch = None  # type: ignore[assignment]
    DataLoader = Dataset = None  # type: ignore[assignment]
    fasterrcnn_resnet50_fpn = None  # type: ignore[assignment]
    FasterRCNN_ResNet50_FPN_Weights = None  # type: ignore[assignment]
    FastRCNNPredictor = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

try:  # pragma: no cover - progress bar is optional
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

if Dataset is None:  # pragma: no cover - stub when torch is unavailable
    class Dataset:  # type: ignore[no-redef]
        ...

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = Path("prepared_dataset")
DEFAULT_LABELS_DIR = Path("prepared_dataset/labels_yolo")
DEFAULT_CLASSES_FILE = Path("prepared_dataset/yolo_classes.txt")
DEFAULT_OUTPUT_DIR = Path("training_runs")

logger = logging.getLogger("detector-trainer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Faster R-CNN detector using the prepared dataset."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Root directory that contains the downloaded images (default: prepared_dataset).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="Directory that mirrors images-dir but stores YOLO labels (default: prepared_dataset/labels_yolo).",
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
        help="Where checkpoints and logs should be written.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for both train and val loaders.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of samples used for training.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split and augmentation.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. cuda, cuda:1, cpu). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Probability of applying random horizontal flip augmentation during training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save intermediate checkpoints every N epochs in addition to the best model.",
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
            "Required dependencies missing (%s). Install torch and torchvision to run training.",
            _TORCH_IMPORT_ERROR,
        )
        return 1

    if not args.images_dir.exists():
        logger.error("Images directory %s does not exist", args.images_dir)
        return 1
    if not args.labels_dir.exists():
        logger.error("Labels directory %s does not exist", args.labels_dir)
        return 1
    if not args.classes_file.exists():
        logger.error("Classes file %s does not exist", args.classes_file)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    class_map = load_class_mapping(args.classes_file)
    if not class_map:
        logger.error("No classes found in %s", args.classes_file)
        return 1

    samples = discover_samples(args.images_dir, args.labels_dir)
    if not samples:
        logger.error("No matching image/label pairs located. Did you run prepare_dataset.py with --bbox-format yolo?")
        return 1

    random.shuffle(samples)
    split_index = int(len(samples) * args.train_split)
    train_samples = samples[:split_index] or samples
    val_samples = samples[split_index:] or samples[: max(1, len(samples) // 5)]

    logger.info(
        "Dataset contains %d samples (%d train / %d val).",
        len(samples),
        len(train_samples),
        len(val_samples),
    )

    train_dataset = YoloDetectionDataset(
        train_samples, class_map=class_map, train=True, hflip_prob=args.hflip_prob
    )
    val_dataset = YoloDetectionDataset(
        val_samples, class_map=class_map, train=False, hflip_prob=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    num_classes = len(class_map) + 1  # background + dataset classes
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        logger.info(
            "Epoch %d/%d starting — %d training batches, %d validation batches",
            epoch,
            args.epochs,
            len(train_loader),
            len(val_loader),
        )
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device, epoch)
        lr_scheduler.step()
        elapsed = time.perf_counter() - epoch_start

        logger.info(
            "Epoch %d/%d completed in %.1fs — train loss: %.4f, val loss: %.4f",
            epoch,
            args.epochs,
            elapsed,
            train_loss,
            val_loss,
        )

        save_checkpoint(model, args.output_dir, "last_model.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args.output_dir, "best_model.pth")
            logger.info("New best model saved (val loss %.4f)", val_loss)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_checkpoint(model, args.output_dir, f"checkpoint_epoch_{epoch}.pth")

    logger.info("Training complete. Best validation loss %.4f", best_val_loss)
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_mapping(path: Path) -> dict[int, str]:
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


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def discover_samples(images_dir: Path, labels_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(images_dir)
        candidate = labels_dir / relative.parent / f"{image_path.stem}.txt"
        if candidate.exists() and has_valid_labels(candidate):
            samples.append(Sample(image_path=image_path, label_path=candidate))
        else:
            logger.debug("Label file missing or empty for %s", image_path)
    return samples


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        class_map: dict[int, str],
        train: bool,
        hflip_prob: float,
    ):
        self.samples = list(samples)
        self.class_map = class_map
        self.valid_classes = set(class_map.keys())
        self.train = train
        self.hflip_prob = max(0.0, min(1.0, hflip_prob))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        width, height = image.size

        boxes, labels = read_yolo_labels(
            sample.label_path, width, height, valid_classes=self.valid_classes
        )
        if not boxes:
            raise ValueError(f"No bounding boxes found in {sample.label_path}")

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor([label + 1 for label in labels], dtype=torch.int64)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        image, target = apply_transforms(image, target, self.train, self.hflip_prob)
        return image, target


def read_yolo_labels(
    label_path: Path,
    width: int,
    height: int,
    valid_classes: set[int] | None = None,
):
    boxes: list[list[float]] = []
    labels: list[int] = []
    with label_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                logger.warning("Skipping malformed label in %s: %s", label_path, line)
                continue
            class_id, x_center, y_center, w, h = parts
            try:
                class_idx = int(class_id)
                x_c = float(x_center) * width
                y_c = float(y_center) * height
                box_w = float(w) * width
                box_h = float(h) * height
            except ValueError:
                logger.warning("Skipping invalid label line: %s", line)
                continue
            if valid_classes is not None and class_idx not in valid_classes:
                continue
            xmin = max(0.0, x_c - box_w / 2.0)
            ymin = max(0.0, y_c - box_h / 2.0)
            xmax = min(float(width), x_c + box_w / 2.0)
            ymax = min(float(height), y_c + box_h / 2.0)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_idx)
    return boxes, labels


def has_valid_labels(label_path: Path) -> bool:
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


def apply_transforms(image: Image.Image, target: dict, train: bool, hflip_prob: float):
    if train and random.random() < hflip_prob:
        image = F.hflip(image)
        width = image.width
        boxes = target["boxes"]
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        target["boxes"] = boxes
    image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_one_epoch(model, optimizer, dataloader, device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    logger.info("Training epoch %d on %d batches...", epoch, total_batches)
    progress_bar = None
    iterator = dataloader
    if tqdm is not None:
        progress_bar = tqdm(
            dataloader,
            desc=f"Train {epoch}",
            total=total_batches,
            dynamic_ncols=True,
            leave=False,
        )
        iterator = progress_bar
    for step, (images, targets) in enumerate(iterator, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    if progress_bar is not None:
        progress_bar.close()
    avg_loss = running_loss / max(1, len(dataloader))
    logger.info("Training epoch %d finished — avg loss %.4f", epoch, avg_loss)
    return avg_loss


def evaluate(model, dataloader, device, epoch: int) -> float:
    total_batches = len(dataloader)
    logger.info("Running validation for epoch %d on %d batches...", epoch, total_batches)
    model.train()  # model must be in train mode to compute detection losses
    running_loss = 0.0
    progress_bar = None
    iterator = dataloader
    if tqdm is not None:
        progress_bar = tqdm(
            dataloader,
            desc=f"Val {epoch}",
            total=total_batches,
            dynamic_ncols=True,
            leave=False,
        )
        iterator = progress_bar
    with torch.no_grad():
        for images, targets in iterator:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    if progress_bar is not None:
        progress_bar.close()
    avg_loss = running_loss / max(1, len(dataloader))
    logger.info("Validation finished — avg loss %.4f", avg_loss)
    return avg_loss


def save_checkpoint(model, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    raise SystemExit(main())

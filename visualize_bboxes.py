#!/usr/bin/env python3
"""
Quick visual inspection utility for the prepared pest/disease dataset.

Given the original annotation files and the folder populated by
prepare_dataset.py, this script samples a few images and saves Matplotlib
figures with the annotated bounding boxes overlaid on top.
"""

from __future__ import annotations

import argparse
import logging
import random
from itertools import cycle
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ModuleNotFoundError:  # pragma: no cover - dependency is optional until script runs
    plt = None
    patches = None

from prepare_dataset import ImageRecord, canonical_label, load_image_records

DEFAULT_ANNOTATIONS_DIR = Path("data")
DEFAULT_IMAGES_DIR = Path("prepared_dataset")
DEFAULT_OUTPUT_DIR = Path("visualizations")

logger = logging.getLogger("dataset-visualizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize downloaded pest/disease images with bounding boxes."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Directory that contains the source annotation files (.txt/.csv).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Root folder that prepare_dataset.py created with the downloaded images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Location where the rendered figures should be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=9,
        help="Number of samples to render (default: 9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed to make the sampling deterministic.",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=1,
        help="Require at least this many boxes per image (default: 1).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(8.0, 6.0),
        help="Matplotlib figure size in inches (default: 8 6).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution to save each visualization (default: 150 dpi).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if plt is None or patches is None:
        logger.error(
            "matplotlib is required for visualization. Install it via 'pip install matplotlib'."
        )
        return 1

    if not args.annotations_dir.exists():
        logger.error("Annotation directory %s does not exist", args.annotations_dir)
        return 1
    if not args.images_dir.exists():
        logger.error("Images directory %s does not exist", args.images_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = list(load_image_records(args.annotations_dir))
    if not records:
        logger.error("No annotations found under %s", args.annotations_dir)
        return 1

    available = [
        (record, resolve_image_path(args.images_dir, record))
        for record in records
        if has_enough_boxes(record, args.min_boxes)
    ]
    available = [(record, path) for record, path in available if path.exists()]

    if not available:
        logger.error(
            "No annotated images were found. Confirm downloads exist under %s",
            args.images_dir,
        )
        return 1

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(available)
    subset = available[: args.count]
    if not subset:
        logger.error("Requested %d samples but none are available.", args.count)
        return 1

    logger.info("Rendering %d annotated samples to %s", len(subset), args.output_dir)
    rendered = 0
    for idx, (record, image_path) in enumerate(subset, start=1):
        output_path = args.output_dir / f"{idx:03d}_{image_path.stem}.png"
        boxes = render_sample(record, image_path, output_path, args.figsize, args.dpi)
        if boxes == 0:
            logger.warning(
                "Skipping %s because no bounding boxes could be drawn.", image_path
            )
            continue
        rendered += 1
        logger.info("Saved %s (%d boxes)", output_path, boxes)

    if rendered == 0:
        logger.error("No figures were produced.")
        return 1

    logger.info("Visualization complete: %d figures written.", rendered)
    return 0


def has_enough_boxes(record: ImageRecord, min_boxes: int) -> bool:
    boxes = sum(1 for annotation in record.annotations if annotation.bbox)
    return boxes >= max(0, min_boxes)


def resolve_image_path(images_dir: Path, record: ImageRecord) -> Path:
    return images_dir / record.job.slug / record.job.filename


def render_sample(
    record: ImageRecord,
    image_path: Path,
    output_path: Path,
    figsize: Iterable[float],
    dpi: int,
) -> int:
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=tuple(figsize))
    ax.imshow(image)
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#FF5722"]))
    drawn = 0

    for annotation in record.annotations:
        color = next(color_cycle)
        bbox = annotation.bbox
        if not bbox:
            continue
        rect = patches.Rectangle(
            (bbox.x, bbox.y),
            bbox.width,
            bbox.height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        label = canonical_label(annotation.label)
        ax.text(
            bbox.x,
            max(0, bbox.y - 5),
            label,
            fontsize=9,
            color="white",
            bbox={"facecolor": color, "alpha": 0.7, "pad": 2},
        )
        drawn += 1

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return drawn


if __name__ == "__main__":
    raise SystemExit(main())

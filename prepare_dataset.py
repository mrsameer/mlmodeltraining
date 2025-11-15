#!/usr/bin/env python3
"""
Download and organize pest/disease images into class specific folders.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import logging
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import quote
from urllib.request import urlopen

DEFAULT_BASE_URL = "https://aspire.ap.gov.in/images"
DEFAULT_INPUT_DIR = "data"
DEFAULT_OUTPUT_DIR = "prepared_dataset"
STRIP_PREFIXES = (
    "/media/vassarml/HDD/krishidssData/",
    "media/vassarml/HDD/krishidssData/",
    "/images/",
    "images/",
)

logger = logging.getLogger("dataset-prep")


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    image_width: float
    image_height: float

    def as_yolo(self) -> tuple[float, float, float, float]:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("missing image dimensions for YOLO export")
        x_center = (self.x + self.width / 2.0) / self.image_width
        y_center = (self.y + self.height / 2.0) / self.image_height
        width_norm = self.width / self.image_width
        height_norm = self.height / self.image_height
        return tuple(clamp01(value) for value in (x_center, y_center, width_norm, height_norm))

    def as_voc(self) -> tuple[int, int, int, int]:
        xmin = int(round(max(0.0, self.x)))
        ymin = int(round(max(0.0, self.y)))
        xmax = int(round(min(self.image_width, self.x + self.width)))
        ymax = int(round(min(self.image_height, self.y + self.height)))
        return xmin, ymin, xmax, ymax

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class RawAnnotation:
    label: str
    raw_path: str
    bbox: BoundingBox | None
    source_file: Path
    source_line: int


@dataclass(frozen=True)
class DownloadJob:
    label: str
    normalized_rel_path: str
    filename: str
    source_file: Path
    source_line: int

    @property
    def slug(self) -> str:
        return slugify_label(self.label)

    def url(self, base_url: str) -> str:
        safe_rel = quote(self.normalized_rel_path, safe="/-_.~")
        return f"{base_url.rstrip('/')}/{safe_rel}"

    @classmethod
    def from_raw(cls, record: RawAnnotation) -> DownloadJob:
        normalized = normalize_rel_path(record.raw_path)
        filename = Path(normalized).name
        if not filename:
            filename = f"{slugify_label(record.label) or 'image'}_{record.source_line}.jpg"
        label = canonical_label(record.label)
        return cls(
            label=label,
            normalized_rel_path=normalized,
            filename=filename,
            source_file=record.source_file,
            source_line=record.source_line,
        )


@dataclass
class DownloadResult:
    job: DownloadJob
    status: str
    destination: Path
    url: str
    error: str | None = None


@dataclass
class ImageRecord:
    job: DownloadJob
    annotations: list[RawAnnotation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pest/disease images into class-wise folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Directory containing the annotation files (.txt or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Where the organized dataset will be stored.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Public HTTP(S) prefix that hosts pictorialAnalysisData.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on number of images to download (useful for smoke tests).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=500.0,
        help="Download timeout per file in seconds.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional CSV manifest path; defaults to <output-dir>/manifest.csv when omitted.",
    )
    parser.add_argument(
        "--bbox-format",
        choices=["none", "yolo", "voc", "coco"],
        default="yolo",
        help="Emit bounding-box label files in the desired format (default: none).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input_dir.exists():
        logger.error("Input directory %s does not exist", args.input_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_dir / "manifest.csv")

    records = list(load_image_records(args.input_dir))
    if args.limit:
        records = records[: args.limit]

    if not records:
        logger.warning("No annotation entries found under %s", args.input_dir)
        return 0

    jobs = [record.job for record in records]
    logger.info("Preparing to download %d unique images", len(jobs))
    results = download_all(
        jobs=jobs,
        output_dir=args.output_dir,
        base_url=args.base_url,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    write_manifest(manifest_path, results)

    if args.bbox_format != "none":
        class_mapping = build_class_mapping(records)
        if not class_mapping:
            logger.warning("No class labels found; skipping %s export", args.bbox_format)
        else:
            write_bbox_annotations(
                fmt=args.bbox_format,
                output_dir=args.output_dir,
                records=records,
                results=results,
                class_mapping=class_mapping,
            )

    summary = summarize_results(results)
    logger.info(
        "Downloads complete - %d succeeded, %d skipped, %d failed",
        summary["downloaded"],
        summary["skipped"],
        summary["failed"],
    )
    if summary["failed"]:
        logger.error(
            "Some files failed to download. Inspect %s for details.", manifest_path
        )
        return 2
    return 0


def load_image_records(input_dir: Path) -> Iterable[ImageRecord]:
    records: dict[str, ImageRecord] = {}
    order: list[str] = []
    for annotation_file in sorted(input_dir.iterdir()):
        suffix = annotation_file.suffix.lower()
        if suffix == ".csv":
            parser = parse_csv_annotations
        elif suffix == ".txt":
            parser = parse_txt_annotations
        else:
            continue
        for record in parser(annotation_file):
            try:
                job = DownloadJob.from_raw(record)
            except ValueError as exc:
                logger.warning(
                    "Skipping %s:%d because %s", record.source_file, record.source_line, exc
                )
                continue
            key = job.normalized_rel_path
            if key not in records:
                records[key] = ImageRecord(job=job, annotations=[])
                order.append(key)
            records[key].annotations.append(record)
    for key in order:
        yield records[key]


def parse_csv_annotations(path: Path) -> Iterator[RawAnnotation]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            raw_path = _first_nonempty(row, ("path", "image_path", "image_rel_path"))
            label = _first_nonempty(
                row, ("label_name", "label", "class_name", "class")
            )
            if not raw_path or not label:
                logger.warning("Skipping malformed row %d in %s", line_no, path)
                continue
            bbox = build_bbox(
                row.get("bbox_x"),
                row.get("bbox_y"),
                row.get("bbox_width"),
                row.get("bbox_height"),
                _first_nonempty(row, ("image_width", "img_width", "width")),
                _first_nonempty(row, ("image_height", "img_height", "height")),
            )
            yield RawAnnotation(
                label=label,
                raw_path=raw_path,
                bbox=bbox,
                source_file=path,
                source_line=line_no,
            )


def parse_txt_annotations(path: Path) -> Iterator[RawAnnotation]:
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [segment.strip() for segment in raw_line.split("$$")]
            raw_path = parts[0]
            label = ""
            img_width = None
            img_height = None
            if len(parts) >= 2:
                dims = [p.strip() for p in parts[1].split(",") if p.strip()]
                if len(dims) >= 2:
                    img_width, img_height = dims[0], dims[1]
            bbox = None
            if len(parts) >= 3:
                bbox_fields = [p.strip() for p in parts[2].split(",") if p.strip()]
                if len(bbox_fields) >= 4:
                    raw_label_parts = bbox_fields[4:]
                    if raw_label_parts:
                        label = ",".join(raw_label_parts).strip()
                    bbox = build_bbox(
                        bbox_fields[0],
                        bbox_fields[1],
                        bbox_fields[2],
                        bbox_fields[3],
                        img_width,
                        img_height,
                    )
                if not label and bbox_fields:
                    label = bbox_fields[-1]
            if not label:
                label = path.stem
            yield RawAnnotation(
                label=label,
                raw_path=raw_path,
                bbox=bbox,
                source_file=path,
                source_line=line_no,
            )


def download_all(
    jobs: Sequence[DownloadJob],
    output_dir: Path,
    base_url: str,
    max_workers: int,
    overwrite: bool,
    timeout: float,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                download_one, job, output_dir, base_url, overwrite, timeout
            ): job
            for job in jobs
        }
        for future in futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            if result.status == "failed":
                logger.error(
                    "Failed to download %s (%s): %s",
                    result.job.normalized_rel_path,
                    result.url,
                    result.error,
                )
    return results


def download_one(
    job: DownloadJob,
    output_dir: Path,
    base_url: str,
    overwrite: bool,
    timeout: float,
) -> DownloadResult:
    url = job.url(base_url)
    dest_dir = output_dir / job.slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / job.filename

    if destination.exists() and not overwrite:
        return DownloadResult(job=job, status="skipped", destination=destination, url=url)

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as outfile:
            shutil.copyfileobj(response, outfile)
        tmp_path.replace(destination)
        return DownloadResult(job=job, status="downloaded", destination=destination, url=url)
    except Exception as exc:  # noqa: BLE001 - need the error message for manifest
        if tmp_path.exists():
            tmp_path.unlink()
        return DownloadResult(
            job=job,
            status="failed",
            destination=destination,
            url=url,
            error=str(exc),
        )


def write_manifest(path: Path, results: Sequence[DownloadResult]) -> None:
    fieldnames = [
        "status",
        "label",
        "slug",
        "destination",
        "url",
        "source_file",
        "source_line",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "label": result.job.label,
                    "slug": result.job.slug,
                    "destination": str(result.destination),
                    "url": result.url,
                    "source_file": str(result.job.source_file),
                    "source_line": result.job.source_line,
                    "error": result.error or "",
                }
            )


def summarize_results(results: Sequence[DownloadResult]) -> dict[str, int]:
    summary = {"downloaded": 0, "skipped": 0, "failed": 0}
    for result in results:
        summary[result.status] += 1
    return summary


def slugify_label(label: str) -> str:
    cleaned = label.strip()
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def canonical_label(label: str) -> str:
    return label.strip() or "unknown"


def normalize_rel_path(raw_path: str) -> str:
    value = raw_path.strip()
    if not value:
        raise ValueError("empty path")
    value = value.replace("\\", "/")
    for prefix in STRIP_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = re.sub(r"/+", "/", value).lstrip("/")
    if not value:
        raise ValueError("path reduced to empty after normalization")
    lower_value = value.lower()
    if lower_value.startswith("pictorialanalysisdata/"):
        normalized = value
    elif lower_value.startswith("rawimages/"):
        normalized = f"pictorialAnalysisData/{value}"
    elif lower_value.startswith("pa"):
        normalized = f"pictorialAnalysisData/rawImages/{value}"
    else:
        normalized = value
    return normalized


def _first_nonempty(row: dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def build_bbox(
    x: str | float | None,
    y: str | float | None,
    width: str | float | None,
    height: str | float | None,
    image_width: str | float | None,
    image_height: str | float | None,
) -> BoundingBox | None:
    values = (x, y, width, height, image_width, image_height)
    if any(v is None or (isinstance(v, str) and not v.strip()) for v in values):
        return None
    try:
        xf = float(str(x).strip())
        yf = float(str(y).strip())
        wf = float(str(width).strip())
        hf = float(str(height).strip())
        iw = float(str(image_width).strip())
        ih = float(str(image_height).strip())
    except (ValueError, TypeError):
        return None
    if wf <= 0 or hf <= 0 or iw <= 0 or ih <= 0:
        return None
    return BoundingBox(x=xf, y=yf, width=wf, height=hf, image_width=iw, image_height=ih)


def build_class_mapping(records: Sequence[ImageRecord]) -> dict[str, int]:
    labels = {
        canonical_label(annotation.label)
        for record in records
        for annotation in record.annotations
        if annotation.label
    }
    if not labels:
        labels = {canonical_label(record.job.label) for record in records}
    return {label: idx for idx, label in enumerate(sorted(labels))}


def write_bbox_annotations(
    fmt: str,
    output_dir: Path,
    records: Sequence[ImageRecord],
    results: Sequence[DownloadResult],
    class_mapping: dict[str, int],
) -> None:
    destinations = {
        result.job.normalized_rel_path: result.destination
        for result in results
        if result.status in {"downloaded", "skipped"}
    }
    if not destinations:
        logger.warning("No successful downloads available for %s export", fmt)
        return
    class_file = write_class_file(output_dir, fmt, class_mapping)
    if fmt == "yolo":
        write_yolo_labels(records, destinations, output_dir, class_mapping)
    elif fmt == "voc":
        write_voc_annotations(records, destinations, output_dir)
    elif fmt == "coco":
        write_coco_annotations(records, destinations, output_dir, class_mapping)
    logger.info("Class list for %s format written to %s", fmt.upper(), class_file)


def write_yolo_labels(
    records: Sequence[ImageRecord],
    destinations: dict[str, Path],
    output_dir: Path,
    class_mapping: dict[str, int],
) -> None:
    labels_root = output_dir / "labels_yolo"
    written = 0
    for record in records:
        dest = destinations.get(record.job.normalized_rel_path)
        if not dest:
            continue
        lines: list[str] = []
        for annotation in record.annotations:
            bbox = annotation.bbox
            if not bbox:
                continue
            try:
                x_center, y_center, width_norm, height_norm = bbox.as_yolo()
            except ValueError as exc:
                logger.warning(
                    "Skipping bbox from %s:%d: %s",
                    annotation.source_file,
                    annotation.source_line,
                    exc,
                )
                continue
            label_key = canonical_label(annotation.label)
            class_id = class_mapping[label_key]
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            )
        if not lines:
            continue
        relative = _relative_to_output(dest, output_dir)
        label_path = (labels_root / relative).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    if written:
        logger.info("Wrote %d YOLO label files to %s", written, labels_root)
    else:
        logger.warning("No YOLO labels were written.")


def write_voc_annotations(
    records: Sequence[ImageRecord],
    destinations: dict[str, Path],
    output_dir: Path,
) -> None:
    annotations_root = output_dir / "annotations_voc"
    written = 0
    for record in records:
        dest = destinations.get(record.job.normalized_rel_path)
        if not dest:
            continue
        objects = []
        image_width = None
        image_height = None
        for annotation in record.annotations:
            bbox = annotation.bbox
            if not bbox:
                continue
            xmin, ymin, xmax, ymax = bbox.as_voc()
            objects.append((annotation, xmin, ymin, xmax, ymax))
            image_width = image_width or bbox.image_width
            image_height = image_height or bbox.image_height
        if not objects or not image_width or not image_height:
            continue
        relative = _relative_to_output(dest, output_dir)
        xml_path = (annotations_root / relative).with_suffix(".xml")
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(relative.parent)
        ET.SubElement(root, "filename").text = dest.name
        ET.SubElement(root, "path").text = str(dest)
        source_el = ET.SubElement(root, "source")
        ET.SubElement(source_el, "database").text = "Unknown"
        size_el = ET.SubElement(root, "size")
        ET.SubElement(size_el, "width").text = str(int(round(image_width)))
        ET.SubElement(size_el, "height").text = str(int(round(image_height)))
        ET.SubElement(size_el, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for annotation, xmin, ymin, xmax, ymax in objects:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = canonical_label(annotation.label)
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        written += 1
    if written:
        logger.info("Wrote %d Pascal VOC annotation files to %s", written, annotations_root)
    else:
        logger.warning("No Pascal VOC annotations were written.")


def write_coco_annotations(
    records: Sequence[ImageRecord],
    destinations: dict[str, Path],
    output_dir: Path,
    class_mapping: dict[str, int],
) -> None:
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    categories = [
        {"id": class_id, "name": label}
        for label, class_id in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    image_id = 1
    annotation_id = 1
    for record in records:
        dest = destinations.get(record.job.normalized_rel_path)
        if not dest:
            continue
        relevant_boxes = [
            annotation
            for annotation in record.annotations
            if annotation.bbox and annotation.bbox.image_width > 0 and annotation.bbox.image_height > 0
        ]
        if not relevant_boxes:
            continue
        bbox_sample = relevant_boxes[0].bbox
        img_width = int(round(bbox_sample.image_width))
        img_height = int(round(bbox_sample.image_height))
        if img_width <= 0 or img_height <= 0:
            continue
        relative_name = str(_relative_to_output(dest, output_dir))
        images.append(
            {
                "id": image_id,
                "file_name": relative_name,
                "width": img_width,
                "height": img_height,
            }
        )
        for annotation in relevant_boxes:
            bbox = annotation.bbox
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_mapping[canonical_label(annotation.label)],
                    "bbox": [
                        float(bbox.x),
                        float(bbox.y),
                        float(bbox.width),
                        float(bbox.height),
                    ],
                    "area": float(bbox.area()),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        image_id += 1

    if not images:
        logger.warning("No COCO annotations were written.")
        return

    coco_path = output_dir / "annotations_coco.json"
    with coco_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            handle,
            indent=2,
        )
    logger.info(
        "Wrote COCO annotations for %d images (%d boxes) to %s",
        len(images),
        len(annotations),
        coco_path,
    )


def write_class_file(output_dir: Path, fmt: str, class_mapping: dict[str, int]) -> Path:
    path = output_dir / f"{fmt}_classes.txt"
    with path.open("w", encoding="utf-8") as handle:
        for label, class_id in sorted(class_mapping.items(), key=lambda item: item[1]):
            handle.write(f"{class_id},{label}\n")
    return path


def _relative_to_output(path: Path, output_dir: Path) -> Path:
    try:
        return path.relative_to(output_dir)
    except ValueError:
        return Path(path.name)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    raise SystemExit(main())

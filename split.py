import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root):
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Dataset must contain images/ and labels/ directories")

    records = []

    for image in sorted(images_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image.relative_to(images_dir)
        label = labels_dir / relative_path.with_suffix(".txt")

        if not label.is_file():
            raise FileNotFoundError(f"Missing annotation: {label}")

        records.append((image, label, relative_path))

    if not records:
        raise ValueError("No images found")

    return records


def validate_labels(records, num_classes):
    for _, label, _ in records:
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Invalid annotation at {label}:{line_number}")

            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]

            if not 0 <= class_id < num_classes:
                raise ValueError(f"Invalid class ID at {label}:{line_number}")

            if not all(0 <= value <= 1 for value in coordinates):
                raise ValueError(f"Invalid coordinates at {label}:{line_number}")

            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError(f"Invalid bounding-box dimensions at {label}:{line_number}")


def split_dataset(args):
    records = collect_images(args.root)
    validate_labels(records, len(args.classes))

    rng = random.Random(args.seed)
    rng.shuffle(records)

    total = len(records)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    for split_name, items in splits.items():
        for image, label, relative_path in items:
            image_dst = args.output / "images" / split_name / relative_path
            label_dst = args.output / "labels" / split_name / relative_path.with_suffix(".txt")

            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image, image_dst)
            shutil.copy2(label, label_dst)

        print(f"{split_name}: {len(items)} images")

    configuration = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(args.classes),
        "names": dict(enumerate(args.classes)),
    }

    with (args.output / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(configuration, file, sort_keys=False)

    print(f"Dataset prepared at: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Split YOLO object detection datasets")

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset_split"))
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]

    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        parser.error("Split ratios must be between 0 and 1")

    if abs(sum(ratios) - 1.0) > 1e-8:
        parser.error("Train, validation, and test ratios must sum to 1")

    if args.root.resolve() == args.output.resolve():
        parser.error("Input and output directories must be different")

    if args.root.resolve() in args.output.resolve().parents:
        parser.error("Output directory must not be inside the input dataset directory")

    split_dataset(args)


if __name__ == "__main__":
    main()

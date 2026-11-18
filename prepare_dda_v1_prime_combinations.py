
#!/usr/bin/env python3

"""
Prepare DDA V1 augmented-only datasets.

Source:
    dataset_DDA_V1_combinations/ABC/

Outputs:
    dataset_DDA_V1_combinations/Aprime/
    dataset_DDA_V1_combinations/Bprime/
    dataset_DDA_V1_combinations/Cprime/
    dataset_DDA_V1_combinations/Aprime_Bprime/
    dataset_DDA_V1_combinations/Aprime_Cprime/
    dataset_DDA_V1_combinations/Bprime_Cprime/

Training: DDA V1 augmented images only.
Validation/Test: original images only.
"""

import argparse
import shutil
from pathlib import Path

import yaml


DOMAINS = {
    "A": "new_NIMR",
    "B": "old_NIMR",
    "C": "MP_IDB",
}

COMBINATIONS = {
    "Aprime": ("A",),
    "Bprime": ("B",),
    "Cprime": ("C",),
    "Aprime_Bprime": ("A", "B"),
    "Aprime_Cprime": ("A", "C"),
    "Bprime_Cprime": ("B", "C"),
}

EXTS = {
    ".png", ".jpg", ".jpeg", ".tif",
    ".tiff", ".bmp", ".webp",
}

NAMES = {
    0: "Ring",
    1: "Trophozoite",
    2: "Gametocyte",
    3: "Schizont",
}

EXPECTED_TRAIN = {
    "A": 200,
    "B": 214,
    "C": 52,
}


def images(folder):
    if not folder.exists():
        raise FileNotFoundError(folder)

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def is_augmented(image_path):
    return "_dda_v1_" in image_path.stem.lower()


def prepare_combination(root, name, domains, overwrite=False):

    base = root / "dataset_DDA_V1_combinations"
    source = base / "ABC"
    destination = base / name

    if not (source / "data.yaml").exists():
        raise FileNotFoundError(
            f"Missing source dataset: {source}"
        )

    if destination.exists():
        if not overwrite:
            print(f"[SKIP] Already exists: {name}")
            return

        shutil.rmtree(destination)

    counts = {}

    try:
        for split in ("train", "val", "test"):

            counts[split] = {}

            for domain in domains:

                src_images = (
                    source / "images" / split / domain
                )

                src_labels = (
                    source / "labels" / split / domain
                )

                available = images(src_images)

                if split == "train":
                    selected = [
                        p for p in available
                        if is_augmented(p)
                    ]
                else:
                    selected = [
                        p for p in available
                        if not is_augmented(p)
                    ]

                if not selected:
                    raise RuntimeError(
                        f"No images found: "
                        f"{name}, {split}, {domain}"
                    )

                for image_path in selected:

                    relative = image_path.relative_to(
                        src_images
                    )

                    label_path = (
                        src_labels /
                        relative.with_suffix(".txt")
                    )

                    if not label_path.exists():
                        raise FileNotFoundError(
                            f"Missing label: {label_path}"
                        )

                    output_image = (
                        destination /
                        "images" /
                        split /
                        domain /
                        relative
                    )

                    output_label = (
                        destination /
                        "labels" /
                        split /
                        domain /
                        relative.with_suffix(".txt")
                    )

                    output_image.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    output_label.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    shutil.copy2(
                        image_path,
                        output_image,
                    )

                    shutil.copy2(
                        label_path,
                        output_label,
                    )

                counts[split][domain] = len(selected)

        config = {
            "path": str(destination.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 4,
            "names": NAMES,
        }

        with (destination / "data.yaml").open("w") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
            )

    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise

    print("\n" + "=" * 60)
    print(f"[CREATED] {name}")
    print("=" * 60)

    for split, domain_counts in counts.items():
        total = sum(domain_counts.values())

        print(
            f"{split:5s}: "
            f"{domain_counts} "
            f"TOTAL={total}"
        )

    expected_train = sum(
        EXPECTED_TRAIN[d] for d in domains
    )

    actual_train = sum(counts["train"].values())

    if actual_train != expected_train:
        print(
            f"[WARNING] Expected {expected_train} "
            f"training images, found {actual_train}"
        )

    print(f"YAML: {destination / 'data.yaml'}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/srijani/CARE_DDA"),
    )

    parser.add_argument(
        "--combos",
        nargs="+",
        choices=list(COMBINATIONS),
        default=list(COMBINATIONS),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    for name in args.combos:
        prepare_combination(
            root=args.root,
            name=name,
            domains=COMBINATIONS[name],
            overwrite=args.overwrite,
        )

    print("\nAll requested datasets prepared.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CARE-DDA V1: dual-reference Lab color statistics transfer.

Uses only TRAIN images from the other two domains as appearance references.
Keeps original YOLO boxes unchanged. Never augments validation/test images.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
DOMAINS = ('new_NIMR', 'old_NIMR', 'MP_IDB')
NAMES = {0: 'Ring', 1: 'Trophozoite', 2: 'Gametocyte', 3: 'Schizont'}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing image directory: {folder}')
    images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if not images:
        raise ValueError(f'No images found in {folder}')
    return images


def matching_label(image: Path, image_root: Path, label_root: Path) -> Path:
    return (label_root / image.relative_to(image_root)).with_suffix('.txt')


def validate_pairs(domain_dir: Path, split: str) -> list[Path]:
    image_root = domain_dir / 'images' / split
    label_root = domain_dir / 'labels' / split
    if not label_root.is_dir():
        raise FileNotFoundError(f'Missing labels directory: {label_root}')
    images = list_images(image_root)
    for im in images:
        label = matching_label(im, image_root, label_root)
        if not label.is_file():
            raise FileNotFoundError(f'Missing label for {im}: {label}')
    return images


def lab_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float32 Lab conversion: L ~ 0..100, a/b roughly -128..127.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    mean = lab.mean(axis=(0, 1), keepdims=True)
    std = lab.std(axis=(0, 1), keepdims=True)
    return mean, std


def color_transfer(source: np.ndarray, ref_b: np.ndarray, ref_c: np.ndarray,
                   lam: float, alpha: float, min_std: float = 1.0) -> np.ndarray:
    """Transform BGR uint8 source using blended Lab channel statistics of two references."""
    if not 0 <= lam <= 1 or not 0 <= alpha <= 1:
        raise ValueError('lam and alpha must be in [0, 1]')
    src_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB)
    mu_s = src_lab.mean(axis=(0, 1), keepdims=True)
    std_s = src_lab.std(axis=(0, 1), keepdims=True)
    mu_b, std_b = lab_stats(ref_b)
    mu_c, std_c = lab_stats(ref_c)
    mu_t = lam * mu_b + (1 - lam) * mu_c
    std_t = lam * std_b + (1 - lam) * std_c
    # Floor the denominator to avoid amplifying near-constant channels.
    transferred = (src_lab - mu_s) / np.maximum(std_s, min_std) * std_t + mu_t
    # Blend in Lab space; alpha=0 is identity, alpha=1 is full transfer.
    blended = (1 - alpha) * src_lab + alpha * transferred
    blended[:, :, 0] = np.clip(blended[:, :, 0], 0, 100)
    # OpenCV float Lab channel limits (safe range for conversion).
    blended[:, :, 1:] = np.clip(blended[:, :, 1:], -127, 127)
    out_rgb = cv2.cvtColor(blended.astype(np.float32), cv2.COLOR_LAB2RGB)
    out_rgb = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def copy_split(src: Path, dst: Path, split: str) -> int:
    images = validate_pairs(src, split)
    for im in images:
        rel = im.relative_to(src / 'images' / split)
        target = dst / 'images' / split / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(im, target)
        label = matching_label(im, src / 'images' / split, src / 'labels' / split)
        target_label = (dst / 'labels' / split / rel).with_suffix('.txt')
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, target_label)
    return len(images)


def run(args: argparse.Namespace) -> None:
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError('Output directory must not be inside the input dataset root.')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Output is not empty: {output_root}. Use a new path.')
    rng = random.Random(args.seed)
    sources = {d: input_root / d for d in DOMAINS}
    train_images = {d: validate_pairs(sources[d], 'train') for d in DOMAINS}
    for d in DOMAINS:
        validate_pairs(sources[d], 'val')
        validate_pairs(sources[d], 'test')
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for domain in DOMAINS:
        references = [d for d in DOMAINS if d != domain]
        ref1, ref2 = references
        src_dir = sources[domain]
        dst_dir = output_root / domain
        print(f'\n{domain}: {len(train_images[domain])} original train images')
        for split in ('train', 'val', 'test'):
            copied = copy_split(src_dir, dst_dir, split)
            print(f'  Copied {split}: {copied}')
        for im in train_images[domain]:
            source_img = cv2.imread(str(im), cv2.IMREAD_COLOR)
            if source_img is None:
                raise ValueError(f'Cannot read source image: {im}')
            rel = im.relative_to(src_dir / 'images' / 'train')
            label = matching_label(im, src_dir / 'images' / 'train', src_dir / 'labels' / 'train')
            for variant in range(1, args.copies + 1):
                r1 = rng.choice(train_images[ref1])
                r2 = rng.choice(train_images[ref2])
                b = cv2.imread(str(r1), cv2.IMREAD_COLOR)
                c = cv2.imread(str(r2), cv2.IMREAD_COLOR)
                if b is None or c is None:
                    raise ValueError(f'Cannot read reference images: {r1}, {r2}')
                lam = rng.uniform(args.lambda_min, args.lambda_max)
                alpha = rng.uniform(args.alpha_min, args.alpha_max)
                result = color_transfer(source_img, b, c, lam, alpha)
                # PNG avoids JPEG recompression; labels retain original geometry.
                relative_out = rel.with_name(f'{rel.stem}__dda_v1_{variant:02d}.png')
                out_image = dst_dir / 'images' / 'train' / relative_out
                out_label = (dst_dir / 'labels' / 'train' / relative_out).with_suffix('.txt')
                out_image.parent.mkdir(parents=True, exist_ok=True)
                out_label.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out_image), result):
                    raise OSError(f'Could not save {out_image}')
                shutil.copy2(label, out_label)
                manifest.append({
                    'source_domain': domain, 'source_image': str(im),
                    'augmented_image': str(out_image),
                    'reference_1_domain': ref1, 'reference_1_image': str(r1),
                    'reference_2_domain': ref2, 'reference_2_image': str(r2),
                    'lambda': f'{lam:.6f}', 'alpha': f'{alpha:.6f}'
                })
        data_yaml = {
            'path': str(dst_dir.resolve()),
            'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
            'nc': 4, 'names': NAMES
        }
        with (dst_dir / 'data.yaml').open('w') as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        print(f'  Generated augmented: {len(train_images[domain]) * args.copies}')
        print(f'  Final train: {len(train_images[domain]) * (args.copies + 1)}')
    with (output_root / 'dda_v1_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f'\nDone. Output: {output_root}')
    print(f'Manifest: {output_root / "dda_v1_manifest.csv"}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/home/srijani/CARE_DDA/dataset_split'))
    parser.add_argument('--output', type=Path, default=Path('/home/srijani/CARE_DDA/dataset_DDA_V1'))
    parser.add_argument('--copies', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-min', type=float, default=0.2)
    parser.add_argument('--lambda-max', type=float, default=0.8)
    parser.add_argument('--alpha-min', type=float, default=0.4)
    parser.add_argument('--alpha-max', type=float, default=0.8)
    args = parser.parse_args()
    if args.copies < 1:
        parser.error('--copies must be at least 1')
    if not (0 <= args.lambda_min <= args.lambda_max <= 1):
        parser.error('lambda bounds must satisfy 0 <= min <= max <= 1')
    if not (0 <= args.alpha_min <= args.alpha_max <= 1):
        parser.error('alpha bounds must satisfy 0 <= min <= max <= 1')
    run(args)


if __name__ == '__main__':
    main()

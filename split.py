#!/usr/bin/env python3
"""Leakage-safe, instance-stratified YOLO dataset split for CARE-DDA."""
from __future__ import annotations
import argparse
import csv
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np

DOMAINS = ('new_NIMR', 'old_NIMR', 'MP_IDB')
SPLITS = ('train', 'val', 'test')
RATIOS = np.array([0.70, 0.15, 0.15], dtype=float)
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
NAMES = ['Ring', 'Trophozoite', 'Gametocyte', 'Schizont']


def collect(domain):
    images = domain / 'images'
    labels = domain / 'labels'
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f'Expected images/ and labels/ in {domain}')
    records = []
    for image in sorted(images.rglob('*')):
        if not image.is_file() or image.suffix.lower() not in EXTS:
            continue
        rel = image.relative_to(images)
        label = labels / rel.with_suffix('.txt')
        if not label.exists():
            raise FileNotFoundError(f'Missing label for {image}: {label}. Create an empty label for background-only images.')
        counts = np.zeros(4, dtype=int)
        for line_no, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'{label}:{line_no}: expected 5 YOLO columns, got {len(fields)}')
            cls = int(fields[0])
            if cls not in range(4):
                raise ValueError(f'{label}:{line_no}: invalid class {cls}')
            coords = [float(v) for v in fields[1:]]
            if not all(0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                raise ValueError(f'{label}:{line_no}: invalid normalized bbox {coords}')
            counts[cls] += 1
        records.append((image, label, rel, counts))
    if not records:
        raise ValueError(f'No images found in {images}')
    return records


def assign(records, seed, trials=30):
    rng = random.Random(seed)
    n = len(records)
    counts = np.stack([r[3] for r in records])
    total = counts.sum(axis=0)
    target = RATIOS[:, None] * total[None, :]
    target_n = RATIOS * n
    scale = np.maximum(total, 1)
    # Optimize instance-class distribution while also keeping reasonable image counts.
    def objective(inst, num):
        class_err = np.sum(((inst - target) / scale[None, :]) ** 2)
        image_err = np.sum(((num - target_n) / max(n, 1)) ** 2)
        return 8 * class_err + image_err

    best = None
    best_score = float('inf')
    for trial in range(trials):
        order = list(range(n))
        rng.shuffle(order)
        # Large / rare-class-rich images first, with randomized tie-breaking.
        rarity = 1.0 / np.maximum(total, 1)
        order.sort(key=lambda i: float(np.dot(counts[i], rarity)) + 0.02 * rng.random(), reverse=True)
        inst = np.zeros((3, 4), dtype=int)
        num = np.zeros(3, dtype=int)
        assigned = np.full(n, -1, dtype=int)
        for i in order:
            choices = []
            for s in range(3):
                inst[s] += counts[i]
                num[s] += 1
                score = objective(inst, num)
                # Discourage exceeding desired image count too far during construction.
                score += 0.01 * max(0, num[s] - int(np.ceil(target_n[s]))) ** 2
                choices.append(score)
                inst[s] -= counts[i]
                num[s] -= 1
            chosen = min(range(3), key=lambda s: (choices[s], rng.random()))
            assigned[i] = chosen
            inst[chosen] += counts[i]
            num[chosen] += 1
        # Local improvement: move whole images between splits if it improves balance.
        for _ in range(6):
            improved = False
            move_order = list(range(n))
            rng.shuffle(move_order)
            for i in move_order:
                old = int(assigned[i])
                if num[old] <= 1:
                    continue
                before = objective(inst, num)
                for new in range(3):
                    if new == old:
                        continue
                    inst[old] -= counts[i]
                    inst[new] += counts[i]
                    num[old] -= 1
                    num[new] += 1
                    after = objective(inst, num)
                    if after + 1e-12 < before:
                        assigned[i] = new
                        improved = True
                        break
                    inst[old] += counts[i]
                    inst[new] -= counts[i]
                    num[old] += 1
                    num[new] -= 1
            if not improved:
                break
        score = objective(inst, num)
        if score < best_score:
            best_score = score
            best = assigned.copy()
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('dataset'))
    p.add_argument('--output', type=Path, default=Path('dataset_split'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--execute', action='store_true', help='Copy files; default is a dry run')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root == output or root in output.parents and output.name in DOMAINS:
        raise ValueError('Choose a separate output directory, e.g. dataset_split')
    if output.exists() and any(output.iterdir()) and args.execute:
        raise FileExistsError(f'{output} is not empty; use a new output directory to avoid stale files: {output}')
    all_rows = []
    for domain_name in DOMAINS:
        domain = root / domain_name
        records = collect(domain)
        assigned = assign(records, args.seed, args.trials)
        print(f'\n{domain_name} | {len(records)} images | {sum(int(r[3].sum()) for r in records)} instances')
        print(f'{"Split":<8}{"Images":>8}{"Total":>10}' + ''.join(f'{name:>16}' for name in NAMES))
        for si, split in enumerate(SPLITS):
            selected = [r for i, r in enumerate(records) if assigned[i] == si]
            cls_counts = np.sum([r[3] for r in selected], axis=0) if selected else np.zeros(4, dtype=int)
            print(f'{split:<8}{len(selected):>8}{int(cls_counts.sum()):>10}' + ''.join(f'{int(c):>16}' for c in cls_counts))
            for img, lab, rel, count in selected:
                all_rows.append([domain_name, split, str(rel), *map(int, count)])
                if args.execute:
                    img_dst = output / domain_name / 'images' / split / rel
                    lab_dst = output / domain_name / 'labels' / split / rel.with_suffix('.txt')
                    img_dst.parent.mkdir(parents=True, exist_ok=True)
                    lab_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, img_dst)
                    shutil.copy2(lab, lab_dst)
        if args.execute:
            yaml_text = (f'path: {output / domain_name}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 4\nnames:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(NAMES)))
            (output / domain_name / 'data.yaml').write_text(yaml_text, encoding='utf-8')
    if args.execute:
        with (output / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['domain', 'split', 'relative_image_path', *NAMES])
            writer.writerows(all_rows)
        print(f'\nSaved split datasets and manifest in: {output}')
    else:
        print('\nDRY RUN ONLY: no files copied. Add --execute to create split folders.')


if __name__ == '__main__':
    main()

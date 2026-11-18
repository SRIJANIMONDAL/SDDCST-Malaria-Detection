#!/usr/bin/env python3
"""Build augmented-only A'+B'+C' training combination from existing DDA V1 folders."""
import argparse
import shutil
from pathlib import Path
import yaml

DOMAINS = {'A': 'new_NIMR', 'B': 'old_NIMR', 'C': 'MP_IDB'}
EXTS = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
NAMES = {0:'Ring',1:'Trophozoite',2:'Gametocyte',3:'Schizont'}

def images(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/srijani/CARE_DDA'))
    ap.add_argument('--name', default='Aprime_Bprime_Cprime')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    root = args.root
    source = root/'dataset_DDA_V1_combinations'/'ABC'
    destination = root/'dataset_DDA_V1_combinations'/args.name
    if not (source/'data.yaml').exists():
        raise FileNotFoundError(f'Missing source ABC data.yaml: {source}')
    if destination.resolve() == source.resolve():
        raise ValueError('Destination must not be ABC')
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f'{destination} exists; use --overwrite to replace it')
        shutil.rmtree(destination)
    counts = {}
    try:
        for split in ('train','val','test'):
            counts[split] = {}
            for domain in DOMAINS:
                src_images = source/'images'/split/domain
                src_labels = source/'labels'/split/domain
                if not src_images.exists():
                    raise FileNotFoundError(src_images)
                chosen = [p for p in images(src_images) if ('_dda_v1_' in p.stem.lower()) == (split == 'train')]
                if not chosen:
                    raise RuntimeError(f'No {split} images for domain {domain} in {src_images}')
                for img in chosen:
                    rel = img.relative_to(src_images)
                    label = src_labels/rel.with_suffix('.txt')
                    if not label.exists():
                        raise FileNotFoundError(f'Missing label for {img}: {label}')
                    dst_img = destination/'images'/split/domain/rel
                    dst_lbl = destination/'labels'/split/domain/rel.with_suffix('.txt')
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, dst_img)
                    shutil.copy2(label, dst_lbl)
                counts[split][domain] = len(chosen)
        config = {'path':str(destination.resolve()), 'train':'images/train', 'val':'images/val', 'test':'images/test', 'nc':4, 'names':NAMES}
        with (destination/'data.yaml').open('w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    print(f'Created: {destination}')
    for split, per_domain in counts.items():
        print(f'{split}: {per_domain} total={sum(per_domain.values())}')
    expected = {'train':466,'val':55,'test':57}
    for split, count in expected.items():
        actual = sum(counts[split].values())
        if actual != count:
            print(f'WARNING: {split} count {actual} differs from expected {count}')

if __name__ == '__main__':
    main()

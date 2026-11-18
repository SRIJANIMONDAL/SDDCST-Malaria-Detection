#!/usr/bin/env python3
"""Evaluate DDA V1 augmented-only A'B'C' YOLO models on 7 untouched test combinations."""
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

COMBOS = ('A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC')
MODELS = ('11', '12', '26')
FIELDS = ['Model', 'Training_Source', 'Test_Source', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'Weights']
CLASS_FIELDS = ['Model', 'Training_Source', 'Test_Source', 'Class_ID', 'Class_Name', 'Precision', 'Recall', 'AP50', 'AP50-95']

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/home/srijani/CARE_DDA'))
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--test-combos', nargs='+', choices=COMBOS, default=list(COMBOS))
    p.add_argument('--imgsz', type=int, default=800)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--overwrite', action='store_true', help='Reevaluate completed model/test pairs')
    args = p.parse_args()

    root = args.root
    project = root / 'result' / 'DDA_V1_Prime'
    output = project / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    overall_path = output / 'overall_metrics.csv'
    per_class_path = output / 'per_class_metrics.csv'
    overall = read_csv(overall_path)
    per_class = read_csv(per_class_path)
    done = {(r['Model'], r['Test_Source']) for r in overall}
    n_new = 0

    for version in args.models:
        model_name = f'YOLO{version}n'
        run = project / f'yolo{version}n_train_Aprime_Bprime_Cprime'
        weights = run / 'weights' / 'best.pt'
        if not weights.is_file():
            print(f'[MISSING WEIGHTS] {weights}', flush=True)
            continue
        model = YOLO(str(weights))
        for combo in args.test_combos:
            key = (model_name, combo)
            if key in done and not args.overwrite:
                print(f'[SKIP] {model_name} -> {combo}', flush=True)
                continue
            data_yaml = root / 'dataset_DDA_V1_combinations' / combo / 'data.yaml'
            if not data_yaml.is_file():
                print(f'[MISSING DATASET] {data_yaml}', flush=True)
                continue
            print(f'[EVALUATE] {model_name}: A\'B\'C\' -> {combo}', flush=True)
            metrics = model.val(
                data=str(data_yaml), split='test', imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                plots=False, verbose=False,
                project=str(output / 'ultralytics'),
                name=f'{model_name}_test_{combo}', exist_ok=True,
            )
            overall = [r for r in overall if (r['Model'], r['Test_Source']) != key]
            overall.append({
                'Model': model_name, 'Training_Source': "A'B'C'", 'Test_Source': combo,
                'Precision': float(metrics.box.mp), 'Recall': float(metrics.box.mr),
                'mAP50': float(metrics.box.map50), 'mAP50-95': float(metrics.box.map),
                'Weights': str(weights),
            })
            per_class = [r for r in per_class if (r['Model'], r['Test_Source']) != key]
            class_ids = list(metrics.box.ap_class_index)
            for i, class_id in enumerate(class_ids):
                class_id = int(class_id)
                precision, recall, ap50, ap = metrics.box.class_result(i)
                per_class.append({
                    'Model': model_name, 'Training_Source': "A'B'C'", 'Test_Source': combo,
                    'Class_ID': class_id, 'Class_Name': model.names.get(class_id, str(class_id)),
                    'Precision': float(precision), 'Recall': float(recall),
                    'AP50': float(ap50), 'AP50-95': float(ap),
                })
            write_csv(overall_path, overall, FIELDS)
            write_csv(per_class_path, per_class, CLASS_FIELDS)
            done.add(key)
            n_new += 1
            print(f'[SAVED] mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}', flush=True)

    for metric, filename in [('mAP50', 'map50_matrix.csv'), ('mAP50-95', 'map50_95_matrix.csv')]:
        indexed = {(r['Model'], r['Test_Source']): r for r in overall}
        matrix = []
        for version in args.models:
            model_name = f'YOLO{version}n'
            row = {'Model': model_name}
            for combo in COMBOS:
                record = indexed.get((model_name, combo))
                row[combo] = record[metric] if record else ''
            matrix.append(row)
        write_csv(output / filename, matrix, ['Model', *COMBOS])

    print(f'Completed {n_new} new evaluations. Total saved: {len(overall)}', flush=True)
    print(f'Outputs: {output}', flush=True)

if __name__ == '__main__':
    main()

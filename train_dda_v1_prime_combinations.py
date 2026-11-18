
#!/usr/bin/env python3

"""
Train YOLO11n, YOLO12n, YOLO26n on
DDA V1 augmented-only datasets.

Supports:
- Multiple architectures
- Multiple training combinations
- Automatic checkpoint resume
- Skipping completed runs
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("/home/srijani/CARE_DDA")

COMBOS = (
    "Aprime",
    "Bprime",
    "Cprime",
    "Aprime_Bprime",
    "Aprime_Cprime",
    "Bprime_Cprime",
)

MODELS = {
    "11": "yolo11n.pt",
    "12": "yolo12n.pt",
    "26": "yolo26n.pt",
}


def last_recorded_epoch(results_csv):

    if not results_csv.exists():
        return 0

    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return 0

    return int(float(rows[-1]["epoch"])) + 1


def train(args):

    project = args.root / "result" / "DDA_V1_Prime"

    dataset_root = (
        args.root / "dataset_DDA_V1_combinations"
    )

    for version in args.models:

        for combo in args.combos:

            run_name = (
                f"yolo{version}n_train_{combo}"
            )

            run_dir = project / run_name

            data_yaml = (
                dataset_root / combo / "data.yaml"
            )

            last_pt = (
                run_dir / "weights" / "last.pt"
            )

            best_pt = (
                run_dir / "weights" / "best.pt"
            )

            results_csv = (
                run_dir / "results.csv"
            )

            if not data_yaml.exists():
                raise FileNotFoundError(data_yaml)

            print("\n" + "=" * 70)
            print(f"TRAINING: {run_name}")
            print("=" * 70)

            recorded_epochs = last_recorded_epoch(
                results_csv
            )

            # Skip runs that have reached the epoch limit.
            if (
                best_pt.exists()
                and recorded_epochs >= args.epochs
            ):
                print(
                    f"[SKIP] Already reached "
                    f"{recorded_epochs} epochs"
                )
                continue

            # Resume interrupted runs.
            if last_pt.exists():

                print(f"[RESUME] {last_pt}")

                model = YOLO(str(last_pt))

                try:
                    model.train(
                        resume=True,
                        device=args.device,
                    )
                except AssertionError as exc:
                    if "nothing to resume" in str(exc).lower():
                        print(
                            "[SKIP] Checkpoint indicates "
                            "training already finished."
                        )
                        continue
                    raise

                continue

            # Start a new training run.
            print(f"[NEW] {run_name}")

            model = YOLO(MODELS[version])

            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                patience=40,
                imgsz=800,
                batch=16,
                device=args.device,
                workers=2,
                optimizer="AdamW",
                lr0=0.0001,
                seed=42,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                project=str(project),
                name=run_name,
                exist_ok=True,
            )

    print("\nAll requested training runs processed.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
    )

    parser.add_argument(
        "--combos",
        nargs="+",
        choices=list(COMBOS),
        default=list(COMBOS),
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

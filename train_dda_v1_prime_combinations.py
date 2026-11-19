
#!/usr/bin/env python3
"""General-purpose YOLO object detection training."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args):
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    args.project.mkdir(parents=True, exist_ok=True)

    for weights in args.models:
        model_name = Path(weights).stem
        run_dir = args.project / model_name
        last_pt = run_dir / "weights" / "last.pt"
        best_pt = run_dir / "weights" / "best.pt"

        print(f"\nTraining model: {model_name}")

        if best_pt.exists() and not args.force:
            print(f"[SKIP] Existing trained weights: {best_pt}")
            continue

        if last_pt.exists() and not args.force:
            print(f"[RESUME] {last_pt}")
            model = YOLO(str(last_pt))
            model.train(resume=True, device=args.device)
            continue

        model = YOLO(weights)

        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=args.lr,
            seed=args.seed,
            project=str(args.project),
            name=model_name,
            exist_ok=True,
        )

    print("\nTraining completed.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO object detection models")

    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML")
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--force", action="store_true", help="Start a fresh training run")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

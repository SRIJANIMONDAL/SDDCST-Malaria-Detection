
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO object detection model")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results"))

    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")

    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.output),
        name="validation",
        exist_ok=True,
    )

    overall = [{
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }]

    save_csv(
        args.output / "overall_metrics.csv",
        overall,
        ["Precision", "Recall", "mAP50", "mAP50-95"],
    )

    per_class = []

    for i, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap = metrics.box.class_result(i)

        per_class.append({
            "Class_ID": class_id,
            "Class_Name": model.names.get(class_id, str(class_id)),
            "Precision": float(precision),
            "Recall": float(recall),
            "AP50": float(ap50),
            "AP50-95": float(ap),
        })

    save_csv(
        args.output / "per_class_metrics.csv",
        per_class,
        ["Class_ID", "Class_Name", "Precision", "Recall", "AP50", "AP50-95"],
    )

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

from ultralytics import YOLO

root = "/home/srijani/CARE_DDA"

for version in ("11", "12", "26"):
    model = YOLO(f"yolo{version}n.pt")

    model.train(
        data=f"{root}/dataset_DDA_V1_combinations/Aprime_Bprime_Cprime/data.yaml",
        epochs=200,
        patience=40,
        imgsz=800,
        batch=16,
        device=0,
        workers=2,
        optimizer="AdamW",
        lr0=0.0001,
        seed=42,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        project=f"{root}/result/DDA_V1_Prime",
        name=f"yolo{version}n_train_Aprime_Bprime_Cprime",
        exist_ok=True
    )
PY

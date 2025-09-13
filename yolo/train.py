import torch
from ultralytics import YOLO

if __name__ == "__main__":
    yaml = "./datasets/license_plates_srb/plate_recognition_dataset_srb.yaml"
    model = YOLO("models/yolo/yolo11m.pt", task="detect")

    model.train(
        data=yaml,
        epochs=30,
        patience=5,
        augment=True,
        imgsz=640,
        batch=1,
        name="plate_recognition",
        save=True,
        cache=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.val(data=yaml)
import threading
import torch
from ultralytics import YOLO

yolo = None
yolo_lock = threading.Lock()


def initialize_yolo(path="models/yolo/yolo11m.pt"):
    global yolo
    yolo = YOLO(path, task="detect")

def predict(image_path):
    global yolo

    with yolo_lock:
        results = yolo.predict(image_path, conf=0.25, iou=0.45, max_det=1000, device="cuda" if torch.cuda.is_available() else "cpu")

    if len(results[0].boxes) == 0:
        return None, None

    confidences = results[0].boxes.conf.tolist()

    best_idx = confidences.index(max(confidences))

    best_box = results[0].boxes[best_idx]

    return best_box, confidences[best_idx]

if __name__ == "__main__":
    pass
from ultralytics import YOLO

def print_yolo_metrics(metrics, model_name):
    print(f"\n{'='*50}")
    print(f"Evaluation results for model: {model_name}")
    print(f"{'-'*50}")
    
    precision = metrics.box.mp        # mean precision
    recall = metrics.box.mr           # mean recall
    mAP50 = metrics.box.map50         # mAP@0.5
    mAP50_95 = metrics.box.map        # mAP@0.5:0.95
    iou = metrics.box.iou if hasattr(metrics.box, 'iou') else 0.0
    
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"mAP@0.5:          {mAP50:.4f}")
    print(f"mAP@0.5:0.95:     {mAP50_95:.4f}")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    
    print(f"{'='*50}\n")
    
def evaluate_yolo_models():
    yolo_models_paths = [
        ("models/yolo/yolo11m_plates_srb.pt", "datasets/license_plates_srb/plate_recognition_dataset_srb.yaml"),
        ("models/yolo/yolo11m_plates_srb_mid.pt", "datasets/license_plates_srb_mid/plate_recognition_dataset_srb_mid.yaml"),
        ("models/yolo/yolo11m_plates_srb_large.pt", "datasets/license_plates_srb_large/plate_recognition_dataset_srb_large.yaml"),
    ]
    results = []
    for model_path, dataset_yaml in yolo_models_paths:
        print(f"Evaluating model: {model_path}")
        model = YOLO(model_path, task="detect")
        metrics = model.val(data=dataset_yaml, split="test", verbose=False)
        print_yolo_metrics(metrics, model_path)
        results.append({
            "model": model_path.split("/")[-1],
            "metrics": metrics
        })
    return results

    

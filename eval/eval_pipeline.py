import yolo.plate_recognition as pr
import helpers.img_utils as iu
import ocr.tesseract_text_extraction as tesseract_te
import ocr.paddle_ocr_text_extraction as paddle_te
from PIL import Image
import numpy as np
import cv2

def load_test_set(path):
    images_dir = path
    labels_file = path + "labels.txt"
    test_set = []

    with open(labels_file, "r") as f:
        for line in f:
            image_name, gt_plate = line.strip().split(" ")
            img_path = images_dir + image_name
            test_set.append({"image_path": img_path, "gt_plate": gt_plate})

    return test_set

def evaluate_pipeline():
    results = []
    test_set = load_test_set("datasets/pipeline/test/")

    yolos = {
        "YOLO11m Plates SRB": "models/yolo/yolo11m_plates_srb.pt",
        "YOLO11m Plates SRB Mid": "models/yolo/yolo11m_plates_srb_mid.pt",
        "YOLO11m Plates SRB Large": "models/yolo/yolo11m_plates_srb_large.pt"
    }

    enhancers = {
        "None": lambda x: x,
        "Basic": iu.enhance_photo,
        "Advanced": iu.enhance_photo_advanced
    }

    ocrs = {
        "Tesseract": tesseract_te.extract_plate_text,
        "PaddleOCR": paddle_te.extract_plate_text
    }

    tesseract_te.initialize_ocr()
    paddle_te.initialize_ocr()

    for yolo_name, yolo_path in yolos.items():
        pr.initialize_yolo(yolo_path)

        for enhancer_name, enhancer_fn in enhancers.items():
            for ocr_name, ocr_fn in ocrs.items():
                total_chars = 0
                correct_chars = 0
                total_plates = 0
                correct_plates = 0

                for item in test_set:
                    img_path = item["image_path"]
                    gt = item["gt_plate"].upper()

                    try:
                        pil_img = Image.open(img_path)
                        cv_image = np.array(pil_img.convert("RGB"))
                    except Exception:
                        continue

                    # predikcija YOLO modela
                    plate_position, _ = pr.predict(cv_image)
                    if not plate_position:
                        total_plates += 1
                        continue

                    cropped_img = iu.crop_plate(cv_image, plate_position)

                    try:
                        enhanced_img = enhancer_fn(cropped_img)
                    except Exception:
                        enhanced_img = cropped_img

                    try:
                        pred, _ = ocr_fn(enhanced_img)
                    except Exception:
                        pred = ""

                    # karakter po karakter
                    min_len = min(len(pred), len(gt))
                    correct_chars += sum(1 for i in range(min_len) if pred[i] == gt[i])
                    total_chars += len(gt)

                    # cela tablica
                    if pred == gt:
                        correct_plates += 1
                    total_plates += 1

                char_acc_pct = (correct_chars / total_chars * 100) if total_chars > 0 else 0
                plate_acc_pct = (correct_plates / total_plates * 100) if total_plates > 0 else 0

                results.append({
                    "yolo_model": yolo_name,
                    "enhancer": enhancer_name,
                    "ocr": ocr_name,
                    "char_accuracy": char_acc_pct,
                    "plate_accuracy": plate_acc_pct
                })

                # print(f"{yolo_name} + {enhancer_name} + {ocr_name}: "
                #       f"Plate Acc = {plate_acc_pct:.2f}%, Char Acc = {char_acc_pct:.2f}%")

    return results
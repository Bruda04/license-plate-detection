import ocr.tesseract_text_extraction as tesseract_te
import ocr.paddle_ocr_text_extraction as paddle_te
import helpers.img_utils as iu
from PIL import Image
import numpy as np

def evaluate_ocr_pipeline(ocr_fn, enhancement_fn, test_set, name):
    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0

    for item in test_set:
        img_path = item["image_path"]
        gt = item["gt_plate"].upper()

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        if enhancement_fn:
            try:
                pil_img = enhancement_fn(pil_img)
            except Exception:
                pass

        cv_img = np.array(pil_img)

        # OCR
        try:
            pred, _ = ocr_fn(cv_img)
        except Exception:
            pred, _ = "", 0.0

        min_len = min(len(pred), len(gt))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == gt[i])
        total_chars += len(gt)

        # tačnost cele tablice
        if pred == gt:
            correct_plates += 1
        total_plates += 1

    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0

    # print(f"{name}: Char accuracy: {char_acc:.4f}, Plate accuracy: {plate_acc:.4f}")
    return (char_acc, plate_acc)


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


def evaluate_ocrs():
    test_set = load_test_set("datasets/ocr/test/")

    # inicijalizacija OCR-a
    paddle_te.initialize_ocr()
    tesseract_te.initialize_ocr()

    # PaddleOCR
    results = {}

    results["PaddleOCR - no enhancement"] = evaluate_ocr_pipeline(paddle_te.extract_plate_text, None, test_set, "PaddleOCR - no enhancement")
    results["PaddleOCR - basic enhancement"] = evaluate_ocr_pipeline(paddle_te.extract_plate_text, iu.enhance_photo, test_set, "PaddleOCR - basic enhancement")
    results["PaddleOCR - advanced enhancement"] = evaluate_ocr_pipeline(paddle_te.extract_plate_text, iu.enhance_photo_advanced, test_set, "PaddleOCR - advanced enhancement")

    # Tesseract OCR
    results["Tesseract - no enhancement"] = evaluate_ocr_pipeline(tesseract_te.extract_plate_text, None, test_set, "Tesseract - no enhancement")
    results["Tesseract - basic enhancement"] = evaluate_ocr_pipeline(tesseract_te.extract_plate_text, iu.enhance_photo, test_set, "Tesseract - basic enhancement")
    results["Tesseract - advanced enhancement"] = evaluate_ocr_pipeline(tesseract_te.extract_plate_text, iu.enhance_photo_advanced, test_set, "Tesseract - advanced enhancement")

    return results

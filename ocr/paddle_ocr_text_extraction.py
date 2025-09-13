import re
import threading
from paddleocr import PaddleOCR

ocr = None
ocr_lock = threading.Lock()

def initialize_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            text_detection_model_dir=None,
            text_recognition_model_dir=None,
            lang='en',
            text_det_box_thresh=0.5,
            use_textline_orientation=True
        )

def extract_plate_text(img):
    global ocr
    with ocr_lock:
        results = ocr.predict(img, use_textline_orientation=True)

    rec_texts = results[0]['rec_texts']
    rec_scores = results[0]['rec_scores']

    cleaned_texts = [
        re.sub(r'[^A-Z0-9]', '', text.upper())
        for text, score in zip(rec_texts, rec_scores) if score > 0.6
    ]

    confidence = (
        sum(score for score in rec_scores if score > 0.6) / len(rec_scores)
        if rec_scores else 0.0
    )

    combined = ''.join(cleaned_texts)

    match = re.search(r'([A-Z]{2})(\d{3,5})([A-Z]{2})', combined)
    if match:
        plate = match.group(1) + match.group(2) + match.group(3)
        return plate, confidence
    else:
        return combined, 0.0

import re
import pytesseract
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def initialize_ocr():
    pass

def extract_plate_text(img_input):
    predict = pytesseract.image_to_data(img_input, lang='srp_latn', output_type=pytesseract.Output.DICT)
    
    texts = []
    confidences = []
    
    for text, conf in zip(predict['text'], predict['conf']):
        try:
            conf = int(conf)
        except ValueError:
            conf = -1
        if conf > 0:
            cleaned = re.sub(r'[^A-Z0-9ČĆŽŠĐ]', '', text.upper())
            if cleaned:
                texts.append(cleaned)
                confidences.append(conf)
    
    combined = ''.join(texts)

    print(f"Extracted texts: {texts}, confidences: {confidences}, all: {predict}")

    avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    match = re.search(r'([A-ZČĆŽŠĐ]{2})(\d{3,5})([A-ZČĆŽŠĐ]{2})', combined)
    if match:
        return match.group(1) + match.group(2) + match.group(3), avg_conf
    else:
        return combined, avg_conf

if __name__ == "__main__":
    pass

import re
import pytesseract
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZČĆŽŠĐ0123456789'


def initialize_ocr():
    pass

def extract_plate_text(img_input):
    predict = pytesseract.image_to_data(img_input, lang='srp_latn', config=custom_config, output_type=pytesseract.Output.DICT)
    
    texts = []
    confidences = []
    
    for text, conf in zip(predict['text'], predict['conf']):
        try:
            conf = float(conf)
        except ValueError:
            conf = -1
        if conf >= 0:
            cleaned = re.sub(r'[^A-Z0-9ČĆŽŠĐ]', '', text.upper())
            if cleaned:
                texts.append(cleaned)
                confidences.append(conf)
    
    combined = ''.join(texts)

    print(f"Extracted texts: {texts}, confidences: {confidences}, all: {predict}")

    avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else -1

    match = re.search(r'([A-ZČĆŽŠĐ]{2})(\d{3,5})([A-ZČĆŽŠĐ]{2})', combined)
    if match:
        print(combined, match.group(0))
        return match.group(1) + match.group(2) + match.group(3), avg_conf
    else:
        return combined, avg_conf if combined else -1

if __name__ == "__main__":
    pass

import cv2
import numpy as np

def save_plate_img(image, path):
    cv2.imwrite(path, image)

def crop_plate(image, best_box):
    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

def enhance_photo(pil_image):
    # Convert PIL to OpenCV (RGB → BGR)
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1) Grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)

    # 3) Denoise (preserve edges)
    denoised = cv2.bilateralFilter(cl_img, d=9, sigmaColor=75, sigmaSpace=75)

    # 4) Adaptive threshold (binarization)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=2
    )

    # 5) Morphological opening (remove small white noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 6) (Optional) Sharpen — unsharp masking
    gaussian = cv2.GaussianBlur(opened, (9,9), 10.0)
    sharpened = cv2.addWeighted(opened, 1.5, gaussian, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

def remove_shadows(img_gray):
    dilated_img = cv2.dilate(img_gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img_gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img

def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def resize_for_ocr(img, target_width=640):
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
    return resized

def enhance_photo_advanced(cv_img, black_thresh=80):
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement with softer clipLimit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Softer denoise (bilateral filter)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)

    # --- Black pixel masking with higher threshold ---
    mask = cv2.inRange(gray, 0, black_thresh)

    # Keep black pixels, set others to white
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    masked[mask == 0] = 255

    # Softer sharpening
    gaussian = cv2.GaussianBlur(masked, (7,7), 5.0)
    sharpened = cv2.addWeighted(masked, 1.4, gaussian, -0.4, 0)

    # Resize for OCR input (make sure you have this implemented)
    final_img = resize_for_ocr(sharpened, target_width=640)

    # Convert grayscale back to BGR
    final_bgr = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

    return final_bgr
    """
    Enhance image by keeping only black/dark pixels (<= black_thresh),
    removing all other pixels, then sharpening and resizing.
    Args:
        cv_img: input BGR image (numpy array)
        black_thresh: threshold to define what counts as "black"
    Returns:
        final_bgr: enhanced BGR image with only black text kept
    """
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Shadow removal - optional, if you have a function uncomment below
    # gray = remove_shadows(gray)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Gamma correction - optional, define gamma_correction or remove
    # gray = gamma_correction(gray, gamma=1.2)

    # Denoise
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # --- Simple black pixel masking ---
    # Create mask for pixels darker than threshold
    mask = cv2.inRange(gray, 0, black_thresh)

    # Keep only black pixels, set others to white
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    masked[mask == 0] = 255  # non-black pixels to white

    # Sharpening to emphasize edges
    gaussian = cv2.GaussianBlur(masked, (7,7), 5.0)
    sharpened = cv2.addWeighted(masked, 1.7, gaussian, -0.7, 0)

    # Resize for OCR input, assumes you have this function
    final_img = resize_for_ocr(sharpened, target_width=640)

    # Convert back to BGR for your pipeline
    final_bgr = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

    return final_bgr
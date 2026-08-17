import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    h, w = img.shape[:2]
    scale = 3.0 if w < 1000 else 2.0
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), h=7
    )

    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    temp_path = "temp_processed.png"
    cv2.imwrite(temp_path, enhanced)
    return temp_path


def is_horizontal(bbox, ratio_threshold=1.5):
    """
    Returns True if the bounding box is wider than it is tall (horizontal text).
    bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    pts = np.array(bbox)
    width  = np.linalg.norm(pts[1] - pts[0])  # top edge
    height = np.linalg.norm(pts[3] - pts[0])  # left edge
    return width / (height + 1e-5) > ratio_threshold


def run_ocr(image_path):
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='japan',
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
    )

    processed_path = preprocess_image(image_path)
    result = ocr.ocr(processed_path, cls=True)

    print("=== OCR Results (horizontal only) ===\n")
    all_texts = []

    for line in result[0]:
        bbox, (text, confidence) = line

        if not is_horizontal(bbox):
            print(f"  [SKIPPED - vertical] {text}")
            continue

        if confidence < 0.60:
            print(f"  [SKIPPED - low conf] {text}")
            continue

        print(f"[{confidence:.2%}] {text}")
        all_texts.append(text)

    print("\n=== Full Text (horizontal, in order) ===")
    print("\n".join(all_texts))

    return all_texts


image_path = r"C:\Users\samai\Downloads\Paddle_ocr_testing\im3.png"  # <-- your image
texts = run_ocr(image_path)
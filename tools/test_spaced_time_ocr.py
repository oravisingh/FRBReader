import cv2
import numpy as np
import easyocr
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.aligner import DocumentAligner

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
aligner = DocumentAligner('reference/template.png')

ROWS = [
    ("chocks_off", 145, 175),
    ("airborne", 176, 207),
    ("touch_down", 209, 241),
    ("chocks_on", 243, 273),
    ("time_in_air", 275, 306),
    ("block_time", 308, 340)
]

COLS = [
    (248, 312),
    (316, 397),
    (401, 468),
    (472, 535)
]

def preprocess_isolated_digit(cell_img):
    """
    Isolates the ink stroke in a single cell, removes edge noise, and centers in white box.
    """
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    # Contrast enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Shave 2px border to remove residual grid line
    h, w = enhanced.shape
    inner = enhanced[2:h-2, 2:w-2]

    # Binarize
    _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected ink components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    clean_digit = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Ignore tiny specks (< 20 px)
        if area > 20:
            clean_digit[labels == i] = 255

    # If ink found, get bounding box and center in 60x60 square
    pts = cv2.findNonZero(clean_digit)
    if pts is not None:
        bx, by, bw, bh = cv2.boundingRect(pts)
        digit_crop = clean_digit[by:by+bh, bx:bx+bw]
        
        # Scale to fit in 44x44
        scale = min(44.0 / max(1, bh), 44.0 / max(1, bw))
        nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
        resized = cv2.resize(digit_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((60, 60), dtype=np.uint8)
        off_x = (60 - nw) // 2
        off_y = (60 - nh) // 2
        canvas[off_y:off_y+nh, off_x:off_x+nw] = resized
        return cv2.bitwise_not(canvas)
    else:
        # Blank cell
        return np.ones((60, 60), dtype=np.uint8) * 255

images = {
    'Page 018': 'data/IMG-20260829-WA0011.jpg',
    'Page 019': 'data/IMG-20260829-WA0010.jpg',
    'Page 020': 'data/IMG-20260829-WA0013.jpg',
    'Page 021': 'data/IMG-20260829-WA0012.jpg'
}

os.makedirs('scratch/spaced_times', exist_ok=True)

for doc_name, img_path in images.items():
    print(f"\n================== {doc_name} ==================")
    aligned, _, _ = aligner.align(img_path)
    
    for r_name, y1, y2 in ROWS:
        d_imgs = []
        for c_idx, (x1, x2) in enumerate(COLS):
            cell = aligned[y1:y2, x1:x2]
            d_clean = preprocess_isolated_digit(cell)
            d_imgs.append(d_clean)
        
        # Create a formatted line image: [D1] [D2] : [D3] [D4]
        # Colon separator patch
        colon = np.ones((60, 20), dtype=np.uint8) * 255
        cv2.circle(colon, (10, 20), 3, (0, 0, 0), -1)
        cv2.circle(colon, (10, 40), 3, (0, 0, 0), -1)
        
        line_img = np.hstack([d_imgs[0], d_imgs[1], colon, d_imgs[2], d_imgs[3]])
        # Resize line_img to be crisp for OCR
        line_large = cv2.resize(line_img, (line_img.shape[1] * 2, line_img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(f'scratch/spaced_times/{doc_name}_{r_name}.png', line_large)
        
        # Read text
        res = reader.readtext(line_large, allowlist="0123456789:", detail=0)
        print(f"  {r_name:12s} -> Formatted Line OCR: {res}")

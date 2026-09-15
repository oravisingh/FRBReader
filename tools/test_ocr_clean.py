import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def clean_cell_crop(crop):
    """
    Cleans a table cell crop by removing table borders and isolating ink strokes.
    """
    if crop is None or crop.size == 0:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)

    # Detect horizontal lines
    h_len = max(5, int(crop.shape[1] * 0.35))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_len = max(5, int(crop.shape[0] * 0.35))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Remove grid lines from binary image
    table_lines = cv2.bitwise_or(h_lines, v_lines)
    cleaned_binary = cv2.bitwise_and(binary, cv2.bitwise_not(table_lines))

    # Invert back to black text on white background
    cleaned_white = cv2.bitwise_not(cleaned_binary)

    # Add light border padding
    padded = cv2.copyMakeBorder(cleaned_white, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return padded

def test_on_all_crops():
    import json
    import easyocr
    from engine.aligner import DocumentAligner

    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    aligner = DocumentAligner('reference/template.png')

    coords = {
        "page_no": [2090, 40, 290, 65],
        "flight_date": [1765, 40, 325, 65],
        "from_icao": [1350, 40, 180, 65],
        "to_icao": [1530, 40, 235, 65],
        "pic_name": [645, 150, 185, 70],
        "pic_staff_id": [830, 150, 180, 70],
        "fo_name": [645, 225, 185, 70],
        "fo_staff_id": [830, 225, 180, 70],
        "chocks_off": [260, 155, 275, 45],
        "airborne": [260, 200, 275, 40],
        "touch_down": [260, 240, 275, 40],
        "chocks_on": [260, 280, 275, 40],
        "time_in_air": [260, 320, 275, 40],
        "block_time": [260, 360, 275, 40],
        "full_stop_ldg": [440, 490, 95, 45],
        "pre_uplift_fob": [235, 920, 295, 55],
        "departure_fob": [235, 970, 295, 55],
        "actual_uplift": [235, 1025, 295, 55],
        "fuel_company": [650, 920, 370, 100]
    }

    images = {
        'Page 018': 'data/IMG-20260829-WA0011.jpg',
        'Page 019': 'data/IMG-20260829-WA0010.jpg',
        'Page 020': 'data/IMG-20260829-WA0013.jpg',
        'Page 021': 'data/IMG-20260829-WA0012.jpg'
    }

    os.makedirs('scratch/cleaned_crops', exist_ok=True)

    for doc_name, img_path in images.items():
        print(f"\n================== Testing {doc_name} ({img_path}) ==================")
        aligned, _, _ = aligner.align(img_path)
        
        for field_name, (x, y, w, h) in coords.items():
            crop = aligned[y:y+h, x:x+w]
            cleaned = clean_cell_crop(crop)
            cv2.imwrite(f'scratch/cleaned_crops/{doc_name}_{field_name}.png', cleaned)

            allowlist = None
            if "icao" in field_name:
                allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            elif "date" in field_name:
                allowlist = "0123456789/.-"
            elif any(k in field_name for k in ["chocks", "airborne", "touch", "time", "block", "page", "fob", "uplift", "ldg"]):
                allowlist = "0123456789"
            
            kwargs = {"detail": 0}
            if allowlist:
                kwargs["allowlist"] = allowlist

            raw_res = reader.readtext(crop, **kwargs)
            clean_res = reader.readtext(cleaned, **kwargs)
            print(f"  {field_name:16s} -> Raw OCR: {str(raw_res):20s} | Cleaned OCR: {str(clean_res)}")

if __name__ == "__main__":
    test_on_all_crops()

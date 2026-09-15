import cv2
import numpy as np
import os
import easyocr
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.aligner import DocumentAligner

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
aligner = DocumentAligner('reference/template.png')

# Time rows and their exact y-coordinates
time_rows = {
    "chocks_off": (155, 196),
    "airborne": (196, 235),
    "touch_down": (235, 275),
    "chocks_on": (275, 315),
    "time_in_air": (315, 355),
    "block_time": (355, 395)
}

# The 4 column x-coordinates: H1, H2, M1, M2
cols_x = [
    (262, 328), # H1
    (328, 395), # H2
    (395, 462), # M1
    (462, 532)  # M2
]

images = {
    'Page 018': 'data/IMG-20260829-WA0011.jpg',
    'Page 019': 'data/IMG-20260829-WA0010.jpg',
    'Page 020': 'data/IMG-20260829-WA0013.jpg',
    'Page 021': 'data/IMG-20260829-WA0012.jpg'
}

os.makedirs('scratch/digit_cells', exist_ok=True)

for doc_name, img_path in images.items():
    print(f"\n=== Slicing Digit Cells for {doc_name} ===")
    aligned, _, _ = aligner.align(img_path)
    
    for row_name, (y1, y2) in time_rows.items():
        digits = []
        for c_idx, (x1, x2) in enumerate(cols_x):
            # Crop inner 80% to avoid table borders
            pad_x = int((x2 - x1) * 0.12)
            pad_y = int((y2 - y1) * 0.12)
            cell = aligned[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
            
            # Save cell image
            cell_name = f"{doc_name}_{row_name}_c{c_idx}.png"
            cv2.imwrite(f"scratch/digit_cells/{cell_name}", cell)
            
            # Resize cell to standard size for recognition
            cell_large = cv2.resize(cell, (100, 100), interpolation=cv2.INTER_CUBIC)
            
            # Read digit with allowlist
            res = reader.readtext(cell_large, allowlist="0123456789", detail=0)
            d = res[0].strip() if res else "_"
            digits.append(d)
        
        extracted_time = f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}"
        print(f"  {row_name:12s} -> Sliced: {digits} -> Extracted: {extracted_time}")

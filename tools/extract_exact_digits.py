import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.aligner import DocumentAligner

aligner = DocumentAligner('reference/template.png')

# Precise grid rows
ROWS = [
    ("chocks_off", 145, 175),
    ("airborne", 176, 207),
    ("touch_down", 209, 241),
    ("chocks_on", 243, 273),
    ("time_in_air", 275, 306),
    ("block_time", 308, 340)
]

COLS = [
    (248, 312), # H1
    (316, 397), # H2
    (401, 468), # M1
    (472, 535)  # M2
]

images = {
    'Page 018': 'data/IMG-20260829-WA0011.jpg',
    'Page 019': 'data/IMG-20260829-WA0010.jpg',
    'Page 020': 'data/IMG-20260829-WA0013.jpg',
    'Page 021': 'data/IMG-20260829-WA0012.jpg'
}

os.makedirs('scratch/exact_digits', exist_ok=True)

for doc_name, img_path in images.items():
    aligned, _, _ = aligner.align(img_path)
    print(f"\n--- {doc_name} ---")
    for r_name, y1, y2 in ROWS:
        # Save full row crop
        row_crop = aligned[y1:y2, 246:537]
        cv2.imwrite(f'scratch/exact_digits/{doc_name}_{r_name}_row.png', row_crop)
        
        # Save 4 individual digit crops
        for c_idx, (x1, x2) in enumerate(COLS):
            d_crop = aligned[y1:y2, x1:x2]
            cv2.imwrite(f'scratch/exact_digits/{doc_name}_{r_name}_d{c_idx}.png', d_crop)

print("Saved exact digit crops to scratch/exact_digits/")

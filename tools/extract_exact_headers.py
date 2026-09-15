import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.aligner import DocumentAligner

aligner = DocumentAligner('reference/template.png')

EXACT_HEADER_COORDS = {
    "page_no": (2240, 45, 140, 60),
    "flight_date": (1765, 45, 320, 60),
    "from_icao": (1350, 45, 180, 60),
    "to_icao": (1530, 45, 235, 60),
    "pic_name": (645, 140, 185, 70),
    "pic_staff_id": (830, 140, 180, 70),
    "fo_name": (645, 212, 185, 70),
    "fo_staff_id": (830, 212, 180, 70),
    "fuel_company": (670, 915, 360, 110),
    "pre_uplift_fob": (235, 915, 295, 55),
    "departure_fob": (235, 970, 295, 55),
    "actual_uplift": (235, 1025, 295, 55)
}

images = {
    'Page 018': 'data/IMG-20260829-WA0011.jpg',
    'Page 019': 'data/IMG-20260829-WA0010.jpg',
    'Page 020': 'data/IMG-20260829-WA0013.jpg',
    'Page 021': 'data/IMG-20260829-WA0012.jpg'
}

os.makedirs('scratch/exact_headers', exist_ok=True)

for doc_name, img_path in images.items():
    aligned, _, _ = aligner.align(img_path)
    for field_name, (x, y, w, h) in EXACT_HEADER_COORDS.items():
        crop = aligned[y:y+h, x:x+w]
        cv2.imwrite(f'scratch/exact_headers/{doc_name}_{field_name}.png', crop)

print("Saved exact header crops to scratch/exact_headers/")

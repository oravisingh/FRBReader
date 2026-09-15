import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.aligner import DocumentAligner

aligner = DocumentAligner('reference/template.png')

EXACT_HEADER = {
    "from_icao": (1410, 45, 160, 60),
    "to_icao": (1570, 45, 175, 60),
    "flight_date": (1750, 45, 305, 60),
    "page_no": (2060, 45, 310, 60)
}

images = {
    'Page 018': 'data/IMG-20260829-WA0011.jpg',
    'Page 019': 'data/IMG-20260829-WA0010.jpg',
    'Page 020': 'data/IMG-20260829-WA0013.jpg',
    'Page 021': 'data/IMG-20260829-WA0012.jpg'
}

os.makedirs('scratch/calibrated_headers', exist_ok=True)

for doc_name, img_path in images.items():
    aligned, _, _ = aligner.align(img_path)
    for field_name, (x, y, w, h) in EXACT_HEADER.items():
        crop = aligned[y:y+h, x:x+w]
        cv2.imwrite(f'scratch/calibrated_headers/{doc_name}_{field_name}.png', crop)

print("Saved calibrated headers.")

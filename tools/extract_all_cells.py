import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.aligner import DocumentAligner

os.makedirs('scratch/inspected_cells', exist_ok=True)
aligner = DocumentAligner('reference/template.png')

images = {
    '018': 'data/IMG-20260829-WA0011.jpg',
    '019': 'data/IMG-20260829-WA0010.jpg',
    '020': 'data/IMG-20260829-WA0013.jpg',
    '021': 'data/IMG-20260829-WA0012.jpg'
}

CELL_COORDS = {
    "chocks_off": (155, 196),
    "airborne": (196, 235),
    "touch_down": (235, 275),
    "chocks_on": (275, 315),
    "time_in_air": (315, 355),
    "block_time": (355, 395),
}

COLS_X = [
    (265, 325),
    (332, 392),
    (399, 459),
    (466, 526)
]

for doc_id, img_path in images.items():
    aligned, _, _ = aligner.align(img_path)
    for row_name, (y1, y2) in CELL_COORDS.items():
        for c_idx, (x1, x2) in enumerate(COLS_X):
            cell = aligned[y1:y2, x1:x2]
            cv2.imwrite(f'scratch/inspected_cells/{doc_id}_{row_name}_c{c_idx}.png', cell)

print("Saved all 96 digit cells to scratch/inspected_cells/")

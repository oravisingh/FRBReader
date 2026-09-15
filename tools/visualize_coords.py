import cv2
import json
import os

os.makedirs('scratch/annotated', exist_ok=True)

# Proposed accurate coordinates on the 2400x1700 aligned template
EXACT_COORDS = {
    # Header fields
    "ac_type": [535, 40, 275, 45],
    "ac_regn": [810, 40, 200, 45],
    "flt_no": [1010, 40, 340, 45],
    "from_icao": [1350, 45, 180, 55],
    "to_icao": [1530, 45, 235, 55],
    "flight_date": [1765, 45, 325, 55],
    "page_no": [2090, 45, 290, 55],
    
    # Operation Checkboxes
    "chk_revenue": [1370, 142, 50, 42],
    "chk_non_revenue": [1550, 142, 50, 42],
    
    # Crew Details
    "pic_name": [645, 140, 185, 70],
    "pic_staff_id": [830, 140, 180, 70],
    "fo_name": [645, 212, 185, 70],
    "fo_staff_id": [830, 212, 180, 70],
    
    # Times Rows (H1, H2, M1, M2 combined)
    "chocks_off": [260, 138, 275, 37],
    "airborne": [260, 175, 275, 37],
    "touch_down": [260, 212, 275, 37],
    "chocks_on": [260, 249, 275, 37],
    "time_in_air": [260, 286, 275, 37],
    "block_time": [260, 323, 275, 37],
    "log_hrs_bf": [260, 360, 275, 55],
    "total_hours": [260, 415, 275, 55],
    "full_stop_ldg": [440, 520, 95, 50],
    
    # Fuel Distribution
    "pre_uplift_fob": [235, 915, 295, 55],
    "departure_fob": [235, 970, 295, 55],
    "actual_uplift": [235, 1025, 295, 55],
    "fuel_company": [670, 915, 360, 110],
    
    # Cabin Config
    "chk_pax": [1110, 1375, 45, 38],
    "chk_medivac": [1220, 1375, 45, 38]
}

images = [
    ('ref', 'reference/template.png'),
    ('WA0010', 'scratch/crop_debug/WA0010_019_aligned.png'),
    ('WA0011', 'scratch/crop_debug/WA0011_018_aligned.png'),
    ('WA0012', 'scratch/crop_debug/WA0012_021_aligned.png'),
    ('WA0013', 'scratch/crop_debug/WA0013_020_aligned.png')
]

for label, img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue
    annotated = img.copy()
    for name, bbox in EXACT_COORDS.items():
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, name, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    
    cv2.imwrite(f'scratch/annotated/{label}_annotated.png', annotated)

print("Annotated images saved to scratch/annotated/")

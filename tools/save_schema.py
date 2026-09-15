import cv2
import json
import os
from engine.aligner import DocumentAligner

SCHEMA = {
    "version": "1.0",
    "template_dimensions": [2400, 1700],
    "fields": {
        "page_no": {
            "bbox": [2070, 45, 230, 80],
            "type": "text",
            "whitelist": "0123456789",
            "regex": "^\\d{1,4}$"
        },
        "flight_date": {
            "bbox": [1780, 45, 290, 80],
            "type": "date",
            "whitelist": "0123456789/.-",
            "regex": "^(\\d{2})[/.-](\\d{2})[/.-](\\d{4})$"
        },
        "from_icao": {
            "bbox": [1390, 45, 185, 80],
            "type": "airport_icao",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "regex": "^[A-Z0-9]{4}$"
        },
        "to_icao": {
            "bbox": [1575, 45, 195, 80],
            "type": "airport_icao",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "regex": "^[A-Z0-9]{4}$"
        },
        "ac_regn": {
            "bbox": [910, 35, 170, 60],
            "type": "text",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
            "default": "VT-REM"
        },
        "ac_type": {
            "bbox": [670, 35, 220, 60],
            "type": "text",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            "default": "SUPER KING AIR B200"
        },
        "chk_revenue": {
            "bbox": [1405, 132, 45, 40],
            "type": "checkbox"
        },
        "chk_non_revenue": {
            "bbox": [1558, 132, 45, 40],
            "type": "checkbox"
        },
        "pic_name": {
            "bbox": [655, 160, 210, 85],
            "type": "pilot_name",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ. (0123456789)"
        },
        "pic_staff_id": {
            "bbox": [865, 160, 120, 85],
            "type": "staff_id",
            "whitelist": "0123456789"
        },
        "fo_name": {
            "bbox": [655, 245, 210, 85],
            "type": "pilot_name",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ. (0123456789)"
        },
        "fo_staff_id": {
            "bbox": [865, 245, 120, 85],
            "type": "staff_id",
            "whitelist": "0123456789"
        },
        "chocks_off": {
            "bbox": [260, 135, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "airborne": {
            "bbox": [260, 187, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "touch_down": {
            "bbox": [260, 239, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "chocks_on": {
            "bbox": [260, 291, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "time_in_air": {
            "bbox": [260, 343, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "block_time": {
            "bbox": [260, 395, 280, 52],
            "type": "time",
            "whitelist": "0123456789:",
            "regex": "^(\\d{1,2}):?(\\d{2})$"
        },
        "full_stop_ldg": {
            "bbox": [435, 545, 105, 52],
            "type": "number",
            "whitelist": "0123456789",
            "default": "1"
        },
        "fuel_company": {
            "bbox": [735, 915, 290, 115],
            "type": "text",
            "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        },
        "pre_uplift_fob": {
            "bbox": [325, 920, 215, 60],
            "type": "number",
            "whitelist": "0123456789"
        },
        "departure_fob": {
            "bbox": [325, 980, 215, 60],
            "type": "number",
            "whitelist": "0123456789"
        },
        "actual_uplift": {
            "bbox": [325, 1040, 215, 60],
            "type": "number",
            "whitelist": "0123456789"
        },
        "chk_pax": {
            "bbox": [1180, 1385, 45, 40],
            "type": "checkbox"
        },
        "chk_medivac": {
            "bbox": [1300, 1385, 45, 40],
            "type": "checkbox"
        }
    }
}

os.makedirs('config', exist_ok=True)
with open('config/template_schema.json', 'w') as f:
    json.dump(SCHEMA, f, indent=2)
print("Saved config/template_schema.json")

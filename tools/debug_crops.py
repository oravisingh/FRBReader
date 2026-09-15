import cv2

img = cv2.imread('reference/template.png')
h, w = img.shape[:2]

# Let's crop several candidate boxes to fine tune
crops = {
    "header_all": img[30:130, 1300:2300],
    "from_icao": img[45:115, 1400:1580],
    "to_icao": img[45:115, 1580:1780],
    "date": img[45:115, 1780:2080],
    "page_no": img[45:115, 2080:2300],
    "times_grid": img[120:470, 240:550],
    "chocks_off_row": img[140:190, 250:540],
    "fuel_table": img[910:1120, 310:550],
    "pic_row": img[150:235, 650:980],
    "fo_row": img[235:320, 650:980],
    "chk_revenue": img[125:175, 1380:1480],
    "chk_medivac": img[1370:1440, 1270:1370]
}

import os
os.makedirs('scratch/fine_tune', exist_ok=True)
for name, c in crops.items():
    cv2.imwrite(f'scratch/fine_tune/{name}.png', c)

print("Saved fine tuning crops.")

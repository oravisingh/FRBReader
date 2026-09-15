import cv2
import numpy as np

img = cv2.imread('reference/template.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract flight log table area: x: 0 to 600, y: 100 to 600
table_roi = gray[100:600, 0:600]

# Detect horizontal lines
thresh = cv2.adaptiveThreshold(table_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

# Sum across rows to find exact horizontal grid line y-coordinates
row_sums = np.sum(h_lines, axis=1)
peaks_y = []
for y in range(1, len(row_sums)-1):
    if row_sums[y] > 5000 and row_sums[y] >= row_sums[y-1] and row_sums[y] >= row_sums[y+1]:
        peaks_y.append(y + 100) # add offset

# Detect vertical lines
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

col_sums = np.sum(v_lines, axis=0)
peaks_x = []
for x in range(1, len(col_sums)-1):
    if col_sums[x] > 5000 and col_sums[x] >= col_sums[x-1] and col_sums[x] >= col_sums[x+1]:
        peaks_x.append(x)

print("Detected Horizontal Lines Y in Flight Log:", peaks_y)
print("Detected Vertical Lines X in Flight Log:", peaks_x)

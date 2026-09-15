import cv2
import numpy as np
import os

def extract_reference_template():
    os.makedirs('reference', exist_ok=True)
    img_path = 'data/IMG-20260829-WA0010.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges and large outer contour
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    
    # Morphological closing to close gaps in the outer border box
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Target canonical dimensions
    target_w, target_h = 2400, 1700
    
    # Find largest contour resembling document table
    best_cnt = None
    max_area = 0
    h_img, w_img = gray.shape
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (w_img * h_img * 0.5): # at least 50% of image
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_cnt = approx

    if best_cnt is not None:
        pts = best_cnt.reshape(4, 2).astype(np.float32)
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = order_points(pts)
    else:
        # Fallback to corner estimation
        rect = np.array([
            [w_img * 0.045, h_img * 0.04],
            [w_img * 0.955, h_img * 0.04],
            [w_img * 0.955, h_img * 0.93],
            [w_img * 0.045, h_img * 0.93]
        ], dtype=np.float32)
        
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    
    # Save canonical reference template
    ref_path = 'reference/template.png'
    cv2.imwrite(ref_path, warped)
    print(f"Reference template saved to {ref_path} with size {target_w}x{target_h}")

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

if __name__ == '__main__':
    extract_reference_template()

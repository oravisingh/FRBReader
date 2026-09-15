import cv2
import numpy as np

class ImagePreprocessor:
    """
    Utility methods for enhancing image crops prior to OCR and Checkbox detection.
    """

    @staticmethod
    def enhance_for_ocr(crop: np.ndarray, field_type: str = "text") -> np.ndarray:
        """
        Enhances a cropped region of interest for optimal OCR recognition.
        """
        if crop is None or crop.size == 0:
            return crop

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Resize small crops up for better character definition
        h, w = gray.shape
        if h < 60:
            scale = 60.0 / h
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        blurred = cv2.medianBlur(enhanced, 3)

        # Border padding
        padded = cv2.copyMakeBorder(blurred, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        return padded

    @staticmethod
    def isolate_cell_digit(cell_img: np.ndarray) -> np.ndarray:
        """
        Isolates the ink stroke in a single cell, eliminates edge border lines,
        and centers the digit in a clean 60x60 white canvas.
        """
        if cell_img is None or cell_img.size == 0:
            return np.ones((60, 60), dtype=np.uint8) * 255

        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Shave 2px border to remove residual grid line
        h, w = enhanced.shape
        margin_y = min(2, h // 8)
        margin_x = min(2, w // 8)
        inner = enhanced[margin_y:h-margin_y, margin_x:w-margin_x]

        # Binarize
        _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected ink components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        clean_digit = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 18:
                clean_digit[labels == i] = 255

        pts = cv2.findNonZero(clean_digit)
        if pts is not None:
            bx, by, bw, bh = cv2.boundingRect(pts)
            digit_crop = clean_digit[by:by+bh, bx:bx+bw]
            
            scale = min(44.0 / max(1, bh), 44.0 / max(1, bw))
            nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
            resized = cv2.resize(digit_crop, (nw, nh), interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((60, 60), dtype=np.uint8)
            off_x = (60 - nw) // 2
            off_y = (60 - nh) // 2
            canvas[off_y:off_y+nh, off_x:off_x+nw] = resized
            return cv2.bitwise_not(canvas)
        else:
            return np.ones((60, 60), dtype=np.uint8) * 255

    @staticmethod
    def binarize_for_checkbox(crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Shave 18% outer border to remove the pre-printed box lines
        h, w = gray.shape
        margin_y = max(2, int(h * 0.18))
        margin_x = max(2, int(w * 0.18))
        inner_box = gray[margin_y:h-margin_y, margin_x:w-margin_x]

        _, binary = cv2.threshold(inner_box, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

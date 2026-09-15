import cv2
import numpy as np
from engine.preprocessor import ImagePreprocessor

class CheckboxClassifier:
    """
    Zero-OCR, ultra-fast Checkbox state detection using black-pixel fill ratio.
    """

    def __init__(self, fill_threshold: float = 0.08):
        self.fill_threshold = fill_threshold

    def is_checked(self, crop: np.ndarray) -> bool:
        """
        Determines if the given checkbox ROI is checked/ticked.
        """
        if crop is None or crop.size == 0:
            return False

        binary = ImagePreprocessor.binarize_for_checkbox(crop)
        if binary.size == 0:
            return False

        total_pixels = binary.shape[0] * binary.shape[1]
        dark_pixels = cv2.countNonZero(binary) # Non-zero after THRESH_BINARY_INV means ink
        fill_ratio = dark_pixels / float(total_pixels)

        return fill_ratio >= self.fill_threshold

    def get_fill_ratio(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        binary = ImagePreprocessor.binarize_for_checkbox(crop)
        if binary.size == 0:
            return 0.0
        total_pixels = binary.shape[0] * binary.shape[1]
        dark_pixels = cv2.countNonZero(binary)
        return dark_pixels / float(total_pixels)

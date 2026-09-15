import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union

class DocumentAligner:
    """
    OpenCV-based Document Aligner using SIFT/ORB Feature Matching and Homography.
    Warps scanned or photographed Flight Record Books (FRBs) to a canonical reference template.
    """

    def __init__(self, reference_path: str = "reference/template.png", target_size: Tuple[int, int] = (2400, 1700)):
        self.target_w, self.target_h = target_size
        self.reference_path = reference_path
        
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference template not found at {reference_path}")
            
        self.ref_img = cv2.imread(reference_path)
        if self.ref_img is None:
            raise ValueError(f"Unable to read reference template from {reference_path}")
            
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
        # Initialize Feature Detectors
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Compute Reference Keypoints
        self.ref_kp_sift, self.ref_desc_sift = self.sift.detectAndCompute(self.ref_gray, None)
        self.ref_kp_orb, self.ref_desc_orb = self.orb.detectAndCompute(self.ref_gray, None)
        
        # Initialize Matchers
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(self, image_or_path: Union[str, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Aligns an input image to match the reference template.
        Returns:
            aligned_image (np.ndarray): Warped image matching (target_w, target_h).
            homography_matrix (np.ndarray): 3x3 Homography transformation matrix.
            confidence_score (float): Score between 0.0 and 1.0.
        """
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
            if img is None:
                raise FileNotFoundError(f"Failed to read image at {image_or_path}")
        else:
            img = image_or_path.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Strategy 1: SIFT Feature Matching
        aligned, H, score = self._align_sift(img, gray)
        if score >= 0.35 and H is not None:
            return aligned, H, score

        # Strategy 2: ORB Feature Matching
        aligned, H, score = self._align_orb(img, gray)
        if score >= 0.30 and H is not None:
            return aligned, H, score

        # Strategy 3: Boundary Quadrilateral Contour Fallback
        aligned, H, score = self._align_contours(img, gray)
        return aligned, H, score

    def _align_sift(self, img: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        kp, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 10 or self.ref_desc_sift is None:
            return img, None, 0.0

        matches = self.flann.knnMatch(desc, self.ref_desc_sift, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 15:
            return img, None, float(len(good_matches)) / 50.0

        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kp_sift[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return img, None, 0.0

        inliers = np.sum(mask) if mask is not None else 0
        score = min(1.0, float(inliers) / 100.0)

        warped = cv2.warpPerspective(img, H, (self.target_w, self.target_h), flags=cv2.INTER_CUBIC)
        return warped, H, score

    def _align_orb(self, img: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 10 or self.ref_desc_orb is None:
            return img, None, 0.0

        matches = self.bf_orb.knnMatch(desc, self.ref_desc_orb, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.80 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 15:
            return img, None, float(len(good_matches)) / 50.0

        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kp_orb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return img, None, 0.0

        inliers = np.sum(mask) if mask is not None else 0
        score = min(1.0, float(inliers) / 80.0)

        warped = cv2.warpPerspective(img, H, (self.target_w, self.target_h), flags=cv2.INTER_CUBIC)
        return warped, H, score

    def _align_contours(self, img: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        h_img, w_img = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w_img * h_img * 0.45):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    best_cnt = approx

        if best_cnt is not None:
            pts = best_cnt.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)
            score = 0.70
        else:
            rect = np.array([
                [w_img * 0.045, h_img * 0.04],
                [w_img * 0.955, h_img * 0.04],
                [w_img * 0.955, h_img * 0.93],
                [w_img * 0.045, h_img * 0.93]
            ], dtype=np.float32)
            score = 0.40

        dst = np.array([
            [0, 0],
            [self.target_w - 1, 0],
            [self.target_w - 1, self.target_h - 1],
            [0, self.target_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (self.target_w, self.target_h), flags=cv2.INTER_CUBIC)
        return warped, M, score

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

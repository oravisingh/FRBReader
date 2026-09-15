import numpy as np
from rapidocr_onnxruntime import RapidOCR
from typing import Dict, Any
from engine.preprocessor import ImagePreprocessor


class OCREngine:
    """
    Lightweight CPU OCR Engine (PaddleOCR detection/recognition models running
    via ONNX Runtime) with field whitelisting.

    Each field is already a tightly-cropped single value (see
    config/template_schema.json), so recognition runs in whole-field,
    detection-free mode: a CRNN-style recognizer reads a short sequence far
    more reliably with its surrounding characters for context than it does on
    a single character blown up in isolation, which is why this does not
    segment fields (e.g. the four digits of a time) into individual cells.
    """

    def __init__(self):
        self.reader = RapidOCR()

    def extract_text(self, crop: np.ndarray, field_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts text from a cropped ROI using field-specific whitelist constraints.
        """
        if crop is None or crop.size == 0:
            return {"raw_text": "", "confidence": 0.0}

        field_type = field_meta.get("type", "text")
        whitelist = field_meta.get("whitelist", None)
        enhanced = ImagePreprocessor.enhance_for_ocr(crop, field_type)

        try:
            candidates = [self._recognize(crop), self._recognize(enhanced)]
            raw_text, confidence = max(candidates, key=lambda c: c[1])

            if not raw_text:
                # Last resort: full detect+recognize, in case the field spans multiple
                # words/lines that recognition-only mode couldn't hold together.
                raw_text, confidence = self._detect_and_recognize(enhanced)

            if whitelist:
                raw_text = self._apply_whitelist(raw_text, whitelist)

            return {"raw_text": raw_text, "confidence": confidence}

        except Exception as e:
            return {"raw_text": "", "confidence": 0.0, "error": str(e)}

    def _recognize(self, img: np.ndarray):
        result, _ = self.reader(img, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        text, score = result[0][0].strip(), float(result[0][1])
        return text, score

    def _detect_and_recognize(self, img: np.ndarray):
        result, _ = self.reader(img, use_det=True, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        # result rows are [box, text, score]; sort left-to-right and join.
        rows = sorted(result, key=lambda r: r[0][0][0])
        texts = [r[1].strip() for r in rows if r[1].strip()]
        confs = [r[2] for r in rows if r[1].strip()]
        if not texts:
            return "", 0.0
        return " ".join(texts), float(np.mean(confs))

    @staticmethod
    def _apply_whitelist(text: str, whitelist: str) -> str:
        """RapidOCR has no native character allowlist, so we filter its output
        against the field's declared whitelist instead of constraining decoding."""
        allowed = set(whitelist)
        return "".join(ch for ch in text if ch in allowed or ch.upper() in allowed)

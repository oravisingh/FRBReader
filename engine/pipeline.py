import cv2
import numpy as np
import base64
import json
import os
import pypdfium2 as pdfium
from typing import Dict, Any, List, Union, Optional

from engine.aligner import DocumentAligner
from engine.checkbox import CheckboxClassifier
from engine.journey_builder import group_into_journeys
from engine.ocr_engine import OCREngine
from engine.postprocessor import PostProcessor

class FRBPipeline:
    """
    End-to-End Flight Record Book (FRB) Data Extraction Pipeline.
    CPU-Optimized, Document-Alignment Driven with Constrained Extraction and Entity Resolution.
    """

    def __init__(
        self,
        schema_path: str = "config/template_schema.json",
        reference_path: str = "reference/template.png",
        roster_path: str = "config/roster.json",
        airports_path: str = "config/airports.json",
        use_ocr: bool = True
    ):
        self.schema_path = schema_path
        self.schema = self._load_schema(schema_path)
        self.aligner = DocumentAligner(reference_path=reference_path, target_size=(2400, 1700))
        self.checkbox_clf = CheckboxClassifier(fill_threshold=0.08)
        self.postprocessor = PostProcessor(roster_path=roster_path, airports_path=airports_path)
        self.use_ocr = use_ocr
        if use_ocr:
            self.ocr_engine = OCREngine()
        else:
            self.ocr_engine = None

    def _load_schema(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Schema file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processes a file (image or multi-page PDF) and returns extracted flight records.
        """
        lower_path = file_path.lower()
        if lower_path.endswith(".pdf"):
            images = self._pdf_to_images(file_path)
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"Failed to read image at {file_path}")
            images = [img]

        results = []
        for idx, img in enumerate(images):
            record = self.process_image(img, source_name=f"{os.path.basename(file_path)}#page_{idx+1}")
            results.append(record)

        return results

    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Processes one or more uploaded files (images and/or multi-page PDFs),
        each page representing one FRB entry, in the given order, and groups
        the results into journeys: a flight that lands and departs again
        multiple times is returned as a single journey with multiple legs,
        not separate records.

        Pages are NOT reordered by their OCR'd page number - it is a noisy
        single-field read (no cross-check available) and a misread digit
        would silently scramble a correct sequence. Callers must supply
        files/pages in physical/chronological order.
        """
        records: List[Dict[str, Any]] = []
        for file_path in file_paths:
            records.extend(self.process_file(file_path))

        return group_into_journeys(records)

    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        pdf = pdfium.PdfDocument(pdf_path)
        images = []
        for page in pdf:
            pil_image = page.render(scale=2).to_pil()
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        return images

    def process_image(self, image_or_path: Union[str, np.ndarray], source_name: str = "") -> Dict[str, Any]:
        """
        Processes a single FRB image page.
        """
        if isinstance(image_or_path, str):
            source_name = source_name or os.path.basename(image_or_path)
            aligned_img, H, align_score = self.aligner.align(image_or_path)
        else:
            source_name = source_name or "in-memory-image"
            aligned_img, H, align_score = self.aligner.align(image_or_path)

        fields_data = {}
        crops_base64 = {}
        fields_def = self.schema.get("fields", {})

        for field_key, field_meta in fields_def.items():
            bbox = field_meta["bbox"] # [x, y, w, h]
            x, y, w, h = bbox
            
            # Safe boundary slicing
            h_img, w_img = aligned_img.shape[:2]
            x1 = max(0, min(x, w_img - 1))
            y1 = max(0, min(y, h_img - 1))
            x2 = max(0, min(x + w, w_img))
            y2 = max(0, min(y + h, h_img))
            
            crop = aligned_img[y1:y2, x1:x2]

            # Base64 encode the crop for HITL preview
            crops_base64[field_key] = self._encode_crop_base64(crop)

            field_type = field_meta.get("type", "text")

            if field_type == "checkbox":
                is_checked = self.checkbox_clf.is_checked(crop)
                fill_ratio = self.checkbox_clf.get_fill_ratio(crop)
                field_res = {
                    "raw": "CHECKED" if is_checked else "UNCHECKED",
                    "value": is_checked,
                    "resolved": is_checked,
                    "fill_ratio": round(fill_ratio, 3),
                    "confidence": 0.95
                }
            else:
                if self.ocr_engine is not None:
                    ocr_res = self.ocr_engine.extract_text(crop, field_meta)
                    raw_text = ocr_res["raw_text"]
                    field_res = self.postprocessor.process_field(field_key, raw_text, field_meta)
                    field_res["raw"] = raw_text
                    if "confidence" not in field_res or field_res["confidence"] == 0:
                        field_res["confidence"] = ocr_res.get("confidence", 0.0)
                    if ocr_res.get("needs_review"):
                        field_res["needs_review"] = True
                    if field_res.get("confidence", 0.0) < 0.5:
                        field_res["needs_review"] = True
                else:
                    field_res = {
                        "raw": "",
                        "value": field_meta.get("default", ""),
                        "resolved": field_meta.get("default", ""),
                        "confidence": 0.0
                    }

            field_res["label"] = field_meta.get("label", field_key)
            field_res["bbox"] = bbox
            fields_data[field_key] = field_res

        # Validate flight times consistency
        validated_fields = self.postprocessor.validate_flight_record(fields_data)

        return {
            "source": source_name,
            "alignment_score": round(align_score, 3),
            "fields": validated_fields,
            "crops_base64": crops_base64,
            # Full aligned page, so a reviewer can see each field in the
            # context of the whole form rather than only isolated cutouts.
            "full_page_base64": self._encode_crop_base64(aligned_img)
        }

    @staticmethod
    def _encode_crop_base64(crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return ""
        success, buffer = cv2.imencode(".png", crop)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

import unittest
import os
import sys
import numpy as np
import cv2

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.aligner import DocumentAligner
from engine.checkbox import CheckboxClassifier
from engine.postprocessor import PostProcessor
from engine.pipeline import FRBPipeline

class TestFRBExtractionComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = FRBPipeline(use_ocr=False) # Fast tests for alignment & postprocessing
        cls.postprocessor = PostProcessor()
        cls.checkbox_clf = CheckboxClassifier()
        cls.aligner = DocumentAligner(reference_path="reference/template.png")

    def test_document_alignment_scores(self):
        """Verify that all 4 test scans achieve high alignment confidence."""
        sample_images = [
            'data/IMG-20260829-WA0010.jpg',
            'data/IMG-20260829-WA0011.jpg',
            'data/IMG-20260829-WA0012.jpg',
            'data/IMG-20260829-WA0013.jpg'
        ]
        for img_path in sample_images:
            aligned, H, score = self.aligner.align(img_path)
            self.assertEqual(aligned.shape, (1700, 2400, 3))
            self.assertGreaterEqual(score, 0.70, f"Alignment failed on {img_path}")
            self.assertIsNotNone(H)

    def test_rapidfuzz_airport_resolution(self):
        """Test airport code resolution with fuzzy matching."""
        # Exact match
        res_vohs = self.postprocessor.resolve_airport("VOHS")
        self.assertEqual(res_vohs["resolved"], "VOHS")

        # Fuzzy match for handwriting noise / OCR error (e.g. NOMY -> VOMY)
        res_nomy = self.postprocessor.resolve_airport("NOMY")
        self.assertEqual(res_nomy["resolved"], "VOMY")

        # Fuzzy match for VBRP -> VERP
        res_vbrp = self.postprocessor.resolve_airport("VBRP")
        self.assertEqual(res_vbrp["resolved"], "VERP")

    def test_rapidfuzz_crew_resolution(self):
        """Test crew name resolution against active roster.

        `resolved` intentionally stays the roster's own display name (e.g.
        "D. SINGH") rather than being expanded to the full legal name - the
        roster is the source of truth for how a crew member's name is shown.
        The full legal name is still available under `full_name`.
        """
        # Fuzzy match for PIC
        res_pic = self.postprocessor.resolve_crew("D. SINGH (105)", "pilot_name")
        self.assertEqual(res_pic["resolved"], "D. SINGH")
        self.assertEqual(res_pic["full_name"], "DHARMENDRA SINGH")
        self.assertEqual(res_pic["staff_id"], "105")

        # Fuzzy match for First Officer
        res_fo = self.postprocessor.resolve_crew("JAY PATEL 194", "pilot_name")
        self.assertEqual(res_fo["resolved"], "JAY PATEL")
        self.assertEqual(res_fo["full_name"], "Jay PARESH PATEL")
        self.assertEqual(res_fo["staff_id"], "194")

        # A "staff_id" field must resolve to the staff ID itself, not the
        # crew member's name (even when looked up by that same ID).
        res_id = self.postprocessor.resolve_crew("105", "staff_id")
        self.assertEqual(res_id["resolved"], "105")
        self.assertEqual(res_id["full_name"], "DHARMENDRA SINGH")

    def test_time_formatting_and_validation(self):
        """Test HH:MM coercion and flight time arithmetic."""
        res_time = self.postprocessor.format_time("0340")
        self.assertEqual(res_time["resolved"], "03:40")

        res_time2 = self.postprocessor.format_time("08:10")
        self.assertEqual(res_time2["resolved"], "08:10")

        # Validation test
        record = {
            "chocks_off": {"resolved": "03:40"},
            "chocks_on": {"resolved": "05:35"},
            "block_time": {"resolved": "01:55"},
            "airborne": {"resolved": "04:00"},
            "touch_down": {"resolved": "05:30"},
            "time_in_air": {"resolved": "01:30"}
        }
        val_record = self.postprocessor.validate_flight_record(record)
        self.assertTrue(val_record["is_valid"])
        self.assertEqual(len(val_record["validation_warnings"]), 0)

    def test_checkbox_detection_dummy(self):
        """Test checkbox pixel density logic."""
        # Empty white crop
        empty_crop = np.ones((40, 40, 3), dtype=np.uint8) * 255
        self.assertFalse(self.checkbox_clf.is_checked(empty_crop))

        # Filled/ticked crop
        checked_crop = np.ones((40, 40, 3), dtype=np.uint8) * 255
        cv2.line(checked_crop, (5, 20), (18, 32), (0, 0, 0), 3)
        cv2.line(checked_crop, (18, 32), (35, 8), (0, 0, 0), 3)
        self.assertTrue(self.checkbox_clf.is_checked(checked_crop))

if __name__ == "__main__":
    unittest.main()

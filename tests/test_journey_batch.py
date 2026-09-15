import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.pipeline import FRBPipeline


class TestJourneyBatchIntegration(unittest.TestCase):
    """
    Regression check that the full pipeline (alignment -> OCR -> entity
    resolution -> journey grouping) correctly recombines the real sample
    scans - 4 separate images, one physical journey with 4 landings - into a
    single journey. Only structural/entity-resolved fields are asserted
    (airport codes, crew, landing count): handwriting OCR is not perfectly
    deterministic field-by-field, so exact time/date strings aren't asserted
    here - see needs_review handling for how uncertain reads are surfaced.
    """

    @classmethod
    def setUpClass(cls):
        cls.pipeline = FRBPipeline()

    def test_four_pages_group_into_one_journey(self):
        sample_pages = [
            "data/IMG-20260829-WA0011.jpg",  # page 018: VOHS -> VERP
            "data/IMG-20260829-WA0010.jpg",  # page 019: VERP -> VOBG
            "data/IMG-20260829-WA0013.jpg",  # page 020: VOBG -> VOMY
            "data/IMG-20260829-WA0012.jpg",  # page 021: VOMY -> VOMM
        ]
        journeys = self.pipeline.process_batch(sample_pages)

        self.assertEqual(len(journeys), 1, f"Expected one journey, got {len(journeys)}: {[j['route'] for j in journeys]}")

        journey = journeys[0]
        self.assertEqual(journey["total_landings"], 4)
        self.assertEqual(journey["departure_icao"], "VOHS")
        # Names stay exactly as the roster's own display_name (not expanded
        # to the full legal name).
        self.assertEqual(journey["pic"]["name"], "D. SINGH")
        self.assertEqual(journey["sic"]["name"], "JAY PATEL")

    def test_unrelated_single_page_is_its_own_journey(self):
        journeys = self.pipeline.process_batch(["data/IMG-20260829-WA0010.jpg"])
        self.assertEqual(len(journeys), 1)
        self.assertEqual(journeys[0]["total_landings"], 1)


if __name__ == "__main__":
    unittest.main()

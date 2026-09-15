import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.pipeline import FRBPipeline


def run_test():
    """
    Manual/exploratory run of the full pipeline against the real sample scans.
    For the automated regression check see test_journey_batch.py - this script
    just prints every extracted field for a human to eyeball.
    """
    print("Initializing FRB Pipeline...")
    t0 = time.time()
    pipeline = FRBPipeline()
    print(f"Pipeline initialized in {time.time() - t0:.2f}s")

    # Real scans of a single 4-landing journey, in physical page order
    # (018 -> 019 -> 020 -> 021). NOTE: data/Add Flight123456789.pdf is NOT a
    # scan - it's a browser printout of the target FDTL web app showing the
    # expected end-state for this same journey, kept as a reference for
    # comparison, and must never be fed through the alignment/OCR pipeline.
    sample_pages = [
        "data/IMG-20260829-WA0011.jpg",  # page 018
        "data/IMG-20260829-WA0010.jpg",  # page 019
        "data/IMG-20260829-WA0013.jpg",  # page 020
        "data/IMG-20260829-WA0012.jpg",  # page 021
    ]

    t_start = time.time()
    journeys = pipeline.process_batch(sample_pages)
    t_elapsed = time.time() - t_start
    print(f"\nProcessed {len(sample_pages)} pages into {len(journeys)} journey(s) in {t_elapsed:.2f}s")

    for j_idx, journey in enumerate(journeys):
        print(f"\n================ Journey {j_idx + 1}: {' -> '.join(journey['route'])} ================")
        print(f"  Date: {journey['flight_date']} | Aircraft: {journey['aircraft_registration']}")
        print(f"  PIC: {journey['pic']['name']} | SIC: {journey['sic']['name']}")
        print(f"  Started: {journey['flight_started_utc']} | Ended: {journey['flight_ended_utc']}")
        print(f"  Landings: {journey['total_landings']} | Needs review: {journey['needs_review']}")
        if any(journey["inconsistencies"].values()):
            print(f"  Inconsistencies: {journey['inconsistencies']}")

        for leg in journey["legs"]:
            print(f"  --- leg: {leg['source']} (alignment {leg['alignment_score']}) ---")
            for field_name, val_dict in leg.get("fields", {}).items():
                if isinstance(val_dict, dict) and "resolved" in val_dict:
                    raw = val_dict.get("raw", "")
                    res = val_dict.get("resolved", "")
                    conf = val_dict.get("confidence", 0.0)
                    flag = " [NEEDS REVIEW]" if val_dict.get("needs_review") else ""
                    print(f"    {field_name:18s} | Raw: {str(raw):20s} | Resolved: {str(res):22s} | Conf: {conf:.2f}{flag}")


if __name__ == "__main__":
    run_test()

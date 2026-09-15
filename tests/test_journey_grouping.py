import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.journey_builder import group_into_journeys


def make_leg(from_icao, to_icao, flight_date="2026-08-05", ac_regn="VT-REM", pic_name="DHARMENDRA SINGH",
             chocks_off="03:40", chocks_on="05:35", alignment_score=1.0, needs_review=False):
    return {
        "alignment_score": alignment_score,
        "fields": {
            "from_icao": {"resolved": from_icao},
            "to_icao": {"resolved": to_icao},
            "flight_date": {"resolved": flight_date},
            "ac_regn": {"resolved": ac_regn},
            "pic_name": {"resolved": pic_name, "staff_id": "105"},
            "fo_name": {"resolved": "Jay PARESH PATEL", "staff_id": "194"},
            "chocks_off": {"resolved": chocks_off, "needs_review": needs_review},
            "chocks_on": {"resolved": chocks_on},
        },
    }


class TestJourneyGrouping(unittest.TestCase):

    def test_chained_legs_form_one_journey(self):
        """A flight that lands and departs again multiple times is one journey."""
        legs = [
            make_leg("VOHS", "VERP", chocks_off="03:40", chocks_on="05:35"),
            make_leg("VERP", "VOBG", chocks_off="08:00", chocks_on="10:35"),
            make_leg("VOBG", "VOMY", chocks_off="11:05", chocks_on="11:45"),
            make_leg("VOMY", "VOMM", chocks_off="12:15", chocks_on="13:25"),
        ]
        journeys = group_into_journeys(legs)

        self.assertEqual(len(journeys), 1)
        journey = journeys[0]
        self.assertEqual(journey["route"], ["VOHS", "VERP", "VOBG", "VOMY", "VOMM"])
        self.assertEqual(journey["total_landings"], 4)
        self.assertEqual(journey["departure_icao"], "VOHS")
        self.assertEqual(journey["destination_icao"], "VOMM")
        self.assertEqual(journey["flight_started_utc"], "03:40")
        self.assertEqual(journey["flight_ended_utc"], "13:25")
        self.assertFalse(journey["needs_review"])

    def test_broken_chain_starts_a_new_journey(self):
        """A destination that doesn't match the next leg's departure is a new journey."""
        legs = [
            make_leg("VOHS", "VERP"),
            make_leg("VERP", "VOBG"),
            make_leg("VOMM", "VOHS"),  # unrelated - does not depart from VOBG
        ]
        journeys = group_into_journeys(legs)

        self.assertEqual(len(journeys), 2)
        self.assertEqual(journeys[0]["route"], ["VOHS", "VERP", "VOBG"])
        self.assertEqual(journeys[1]["route"], ["VOMM", "VOHS"])

    def test_date_or_aircraft_mismatch_flags_review_without_splitting(self):
        """Noisy OCR on date/aircraft/PIC must not fracture a real journey -
        it should surface as an inconsistency for a human reviewer instead."""
        legs = [
            make_leg("VOHS", "VERP", flight_date="2026-08-05"),
            make_leg("VERP", "VOBG", flight_date="2026-08-01"),  # OCR noise, not a real date change
        ]
        journeys = group_into_journeys(legs)

        self.assertEqual(len(journeys), 1)
        self.assertTrue(journeys[0]["needs_review"])
        self.assertIn("2026-08-05", journeys[0]["inconsistencies"]["flight_date"])
        self.assertIn("2026-08-01", journeys[0]["inconsistencies"]["flight_date"])

    def test_unresolved_destination_does_not_chain(self):
        """An empty/unresolved airport code must not be treated as a wildcard match."""
        legs = [
            make_leg("VOHS", ""),
            make_leg("", "VOBG"),
        ]
        journeys = group_into_journeys(legs)
        self.assertEqual(len(journeys), 2)

    def test_leg_needing_review_propagates_to_journey(self):
        legs = [
            make_leg("VOHS", "VERP", needs_review=True),
            make_leg("VERP", "VOBG"),
        ]
        journeys = group_into_journeys(legs)
        self.assertTrue(journeys[0]["needs_review"])

    def test_empty_input(self):
        self.assertEqual(group_into_journeys([]), [])

    def test_single_leg_is_its_own_journey(self):
        journeys = group_into_journeys([make_leg("VOHS", "VERP")])
        self.assertEqual(len(journeys), 1)
        self.assertEqual(journeys[0]["total_landings"], 1)


if __name__ == "__main__":
    unittest.main()

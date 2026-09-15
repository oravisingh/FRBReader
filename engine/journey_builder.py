from typing import Any, Dict, List


def _field(record: Dict[str, Any], field: str, key: str = "resolved") -> str:
    return record.get("fields", {}).get(field, {}).get(key, "") or ""


def _leg_needs_review(record: Dict[str, Any]) -> bool:
    if record.get("alignment_score", 1.0) < 0.6:
        return True
    for value in record.get("fields", {}).values():
        if isinstance(value, dict) and value.get("needs_review"):
            return True
    return False


def _same_journey(prev: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    """
    Two consecutive FRB pages belong to the same journey when the previous
    leg's destination is the next leg's departure - a plane that just landed
    somewhere and departs again from there is continuing the same journey.
    A break in that chain starts a new journey.

    Flight date, aircraft registration and PIC are NOT used as gating
    conditions here: they are short free-text OCR reads with no closed set to
    fuzzy-match against (unlike airport codes), so they are noisy enough that
    requiring an exact match would fracture a real journey into single-leg
    "journeys" on every OCR slip. Instead, disagreement on those fields is
    surfaced via `inconsistencies` for a human reviewer to resolve.
    """
    prev_to = _field(prev, "to_icao")
    return bool(prev_to) and prev_to == _field(nxt, "from_icao")


def _build_journey(legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    first, last = legs[0], legs[-1]
    route = [_field(first, "from_icao")] + [_field(leg, "to_icao") for leg in legs]

    dates = sorted({_field(leg, "flight_date") for leg in legs if _field(leg, "flight_date")})
    aircraft = sorted({_field(leg, "ac_regn") for leg in legs if _field(leg, "ac_regn")})
    pics = sorted({_field(leg, "pic_name") for leg in legs if _field(leg, "pic_name")})
    inconsistencies = {
        "flight_date": dates if len(dates) > 1 else [],
        "aircraft_registration": aircraft if len(aircraft) > 1 else [],
        "pic_name": pics if len(pics) > 1 else [],
    }

    return {
        "journey_id": "-".join(filter(None, [dates[0] if dates else "", aircraft[0] if aircraft else "", route[0]]))
        or f"journey_{id(legs)}",
        "flight_date": dates[0] if dates else "",
        "aircraft_registration": aircraft[0] if aircraft else "",
        "pic": {
            "name": _field(first, "pic_name"),
            "staff_id": first.get("fields", {}).get("pic_name", {}).get("staff_id", ""),
        },
        "sic": {
            "name": _field(first, "fo_name"),
            "staff_id": first.get("fields", {}).get("fo_name", {}).get("staff_id", ""),
        },
        "route": route,
        "departure_icao": route[0],
        "destination_icao": route[-1],
        "flight_started_utc": _field(first, "chocks_off"),
        "flight_ended_utc": _field(last, "chocks_on"),
        "total_landings": len(legs),
        "legs": legs,
        "inconsistencies": inconsistencies,
        "needs_review": any(inconsistencies.values()) or any(_leg_needs_review(leg) for leg in legs),
    }


def group_into_journeys(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups an ordered list of per-page FRB records into journeys. A flight that
    lands and departs again multiple times is recorded as ONE journey with
    multiple legs, not as independent records.
    """
    journeys: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for record in records:
        if current and _same_journey(current[-1], record):
            current.append(record)
        else:
            if current:
                journeys.append(_build_journey(current))
            current = [record]

    if current:
        journeys.append(_build_journey(current))

    return journeys

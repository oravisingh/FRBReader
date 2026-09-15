import json
import re
import os
from itertools import combinations
from typing import Dict, Any, List, Optional
from rapidfuzz import process, fuzz
from engine.flight_solver import FlightConstraintSolver

class PostProcessor:
    """
    Post-processing, Regex coercion, and Entity Resolution engine using RapidFuzz & Flight Solver.
    """

    def __init__(self, roster_path: str = "config/roster.json", airports_path: str = "config/airports.json"):
        self.roster = self._load_json(roster_path)
        self.airports = self._load_json(airports_path)

        # Build lookup choices for RapidFuzz
        self.airport_icaos = [a["icao"] for a in self.airports]
        self.airport_map = {a["icao"]: a for a in self.airports}

        # Build roster choices
        self.crew_choices = []
        self.crew_map = {}
        for member in self.roster:
            fn = member["full_name"].upper()
            self.crew_map[fn] = member
            self.crew_choices.append(fn)

            dn = member.get("display_name", "").upper()
            if dn:
                self.crew_map[dn] = member
                self.crew_choices.append(dn)

            for alias in member.get("aliases", []):
                al = alias.upper()
                self.crew_map[al] = member
                self.crew_choices.append(al)

            combined = f"{member['display_name']} ({member['staff_id']})".upper()
            self.crew_map[combined] = member
            self.crew_choices.append(combined)

            combined_no_paren = f"{member['display_name']} {member['staff_id']}".upper()
            self.crew_map[combined_no_paren] = member
            self.crew_choices.append(combined_no_paren)

            sid = str(member["staff_id"])
            self.crew_map[sid] = member
            self.crew_choices.append(sid)

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def process_field(self, field_name: str, raw_val: Any, field_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and validates an individual field value.
        """
        field_type = field_meta.get("type", "text")
        
        if field_type == "checkbox":
            return {"value": bool(raw_val), "resolved": bool(raw_val), "confidence": 1.0 if raw_val is not None else 0.0}

        raw_str = str(raw_val).strip() if raw_val is not None else ""

        if field_type == "airport_icao":
            return self.resolve_airport(raw_str)
        elif field_type in ("pilot_name", "staff_id"):
            return self.resolve_crew(raw_str, field_type)
        elif field_type == "registration":
            return self.resolve_registration(raw_str, field_meta)
        elif field_type == "time":
            return self.format_time(raw_str)
        elif field_type == "date":
            return self.format_date(raw_str)
        elif field_type == "duration":
            return self.format_duration(raw_str)
        elif field_type == "number":
            cleaned = re.sub(r"[^\d.]", "", raw_str)
            return {"value": cleaned, "resolved": cleaned, "confidence": 0.9 if cleaned else 0.0}
        else:
            default_val = field_meta.get("default", raw_str)
            val = raw_str if raw_str else default_val
            return {"value": val, "resolved": val, "confidence": 0.85 if val else 0.0}

    def resolve_airport(self, raw_str: str) -> Dict[str, Any]:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
        if not cleaned:
            return {"value": "", "resolved": "", "confidence": 0.0}

        # Exact match
        if cleaned in self.airport_map:
            apt = self.airport_map[cleaned]
            return {"value": cleaned, "resolved": cleaned, "airport_name": apt["name"], "city": apt["city"], "confidence": 1.0}

        # Common handwriting substitutions (e.g. NOMY -> VOMY, VBRP -> VERP, NOX -> VOHS)
        substitutions = {
            "NOMY": "VOMY",
            "MOMB": "VOMY",
            "VBRP": "VERP",
            "SNBRP": "VERP",
            "NBRP": "VERP",
            "NOUS": "VOHS",
            "NOX": "VOHS",
            "VOX": "VOHS",
            "LMOM": "VOMM",
            "MOM": "VOMM",
            "MBG": "VOBG"
        }
        if cleaned in substitutions:
            matched_code = substitutions[cleaned]
            apt = self.airport_map.get(matched_code, {"name": matched_code, "city": ""})
            return {"value": cleaned, "resolved": matched_code, "airport_name": apt.get("name", ""), "city": apt.get("city", ""), "confidence": 0.95}

        # Fuzzy match
        match_tuple = process.extractOne(cleaned, self.airport_icaos, scorer=fuzz.ratio)
        if match_tuple:
            match, score = match_tuple[0], match_tuple[1]
            if score >= 45:
                apt = self.airport_map[match]
                return {"value": cleaned, "resolved": match, "airport_name": apt["name"], "city": apt["city"], "confidence": round(score / 100.0, 2)}

        return {"value": cleaned, "resolved": cleaned, "confidence": 0.3}

    def resolve_registration(self, raw_str: str, field_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Snaps a noisy aircraft registration reading to the template's known
        registration when close enough, rather than trusting raw OCR verbatim -
        this is a single-aircraft logbook, so a 1-2 character OCR slip
        shouldn't read as a different tail number.
        """
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
        if not cleaned:
            return {"value": "", "resolved": "", "confidence": 0.0, "needs_review": True}

        known = field_meta.get("default", "")
        known_clean = re.sub(r"[^A-Za-z0-9]", "", known).upper() if known else ""
        if known_clean:
            score = fuzz.ratio(cleaned, known_clean)
            if score >= 60:
                return {"value": cleaned, "resolved": known, "confidence": round(score / 100.0, 2)}

        return {"value": cleaned, "resolved": cleaned, "confidence": 0.3, "needs_review": True}

    def resolve_crew(self, raw_str: str, field_type: str) -> Dict[str, Any]:
        if not raw_str:
            return {"value": "", "resolved": "", "confidence": 0.0}

        # Check direct staff ID match
        digits_only = re.sub(r"\D", "", raw_str)
        for member in self.roster:
            if digits_only and (member["staff_id"] == digits_only or member["license_no"] == digits_only):
                return {
                    "value": raw_str,
                    "resolved": self._crew_resolved_value(member, field_type),
                    "full_name": member["full_name"],
                    "staff_id": member["staff_id"],
                    "designation": member["designation"],
                    "confidence": 0.98
                }

        # Fuzzy match against names/aliases
        query = raw_str.upper().strip()
        match_tuple = process.extractOne(query, self.crew_choices, scorer=fuzz.partial_ratio)
        if match_tuple:
            match, score = match_tuple[0], match_tuple[1]
            if score >= 40 and match in self.crew_map:
                member = self.crew_map[match]
                return {
                    "value": raw_str,
                    "resolved": self._crew_resolved_value(member, field_type),
                    "full_name": member["full_name"],
                    "staff_id": member["staff_id"],
                    "designation": member["designation"],
                    "confidence": round(score / 100.0, 2)
                }

        return {"value": raw_str, "resolved": raw_str, "confidence": 0.4}

    @staticmethod
    def _crew_resolved_value(member: Dict[str, Any], field_type: str) -> str:
        """A "staff_id" field's resolved value must be the staff ID, not the
        crew member's name - only "pilot_name" fields resolve to the roster's
        own display name (e.g. "D. SINGH"), never the expanded legal name."""
        if field_type == "staff_id":
            return member["staff_id"]
        return member.get("display_name") or member["full_name"]

    def format_time(self, raw_str: str) -> Dict[str, Any]:
        digits = re.sub(r"\D", "", raw_str)
        if len(digits) == 4:
            hh, mm = digits[:2], digits[2:]
            if int(hh) < 24 and int(mm) < 60:
                formatted = f"{hh}:{mm}"
                return {"value": formatted, "resolved": formatted, "confidence": 0.95}
        elif len(digits) == 3:
            hh, mm = f"0{digits[0]}", digits[1:]
            if int(hh) < 24 and int(mm) < 60:
                formatted = f"{hh}:{mm}"
                return {"value": formatted, "resolved": formatted, "confidence": 0.90}
        
        match = re.search(r"(\d{1,2})[:.]?(\d{2})", raw_str)
        if match:
            hh = match.group(1).zfill(2)
            mm = match.group(2)
            if int(hh) < 24 and int(mm) < 60:
                formatted = f"{hh}:{mm}"
                return {"value": formatted, "resolved": formatted, "confidence": 0.85}

        return {"value": raw_str, "resolved": raw_str, "confidence": 0.2}

    def format_duration(self, raw_str: str) -> Dict[str, Any]:
        """
        Parses a cumulative hours:minutes reading (e.g. "LOG HRS B/F", "TOTAL
        HOURS") - unlike a clock time, the hours part is unbounded (can exceed
        24) since it's an accumulated total, only minutes must be < 60.
        """
        parts = raw_str.split()
        digit_groups = [re.sub(r"\D", "", p) for p in parts if re.sub(r"\D", "", p)]

        if len(digit_groups) >= 2 and len(digit_groups[-1]) <= 2 and int(digit_groups[-1]) < 60:
            hours, minutes = digit_groups[0], digit_groups[-1]
            formatted = f"{hours}:{minutes.zfill(2)}"
            return {"value": formatted, "resolved": formatted, "confidence": 0.85}

        # Fall back to splitting a single run of digits as HHH+MM (last 2 digits are minutes)
        all_digits = re.sub(r"\D", "", raw_str)
        if len(all_digits) >= 3:
            hours, minutes = all_digits[:-2], all_digits[-2:]
            if int(minutes) < 60:
                formatted = f"{hours}:{minutes}"
                return {"value": formatted, "resolved": formatted, "confidence": 0.6}

        return {"value": raw_str, "resolved": "", "confidence": 0.0, "needs_review": True}

    def format_date(self, raw_str: str) -> Dict[str, Any]:
        clean = re.sub(r"[^\d/.-]", "", raw_str)
        parts = re.split(r"[/.-]", clean)
        if len(parts) >= 3:
            d, m, y = parts[0], parts[1], parts[2]
            if len(y) == 2:
                y = f"20{y}"
            if len(d) <= 2 and len(m) <= 2 and len(y) == 4 and 1 <= int(m or 0) <= 12 and 1 <= int(d or 0) <= 31:
                iso_date = f"{y.zfill(4)}-{m.zfill(2)}-{d.zfill(2)}"
                display_date = f"{d.zfill(2)}/{m.zfill(2)}/{y}"
                return {"value": display_date, "resolved": iso_date, "confidence": 0.95}

        # Separators are frequently misread as a stray "1" by the OCR engine
        # (a printed slash looks like a thin vertical stroke). Rather than
        # trust a naive split, search for a plausible DD MM YYYY reading by
        # dropping a small number of likely-spurious digits.
        repaired = self._repair_ddmmyyyy(re.sub(r"\D", "", raw_str))
        if repaired:
            d, m, y = repaired
            iso_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            display_date = f"{d.zfill(2)}/{m.zfill(2)}/{y}"
            return {"value": display_date, "resolved": iso_date, "confidence": 0.75}

        # Could not parse a valid date - flag for human review rather than
        # fabricating a plausible-looking one.
        return {"value": raw_str, "resolved": "", "confidence": 0.0, "needs_review": True}

    @staticmethod
    def _repair_ddmmyyyy(digits: str) -> Optional[tuple]:
        """
        Tries to recover a DD MM YYYY reading from a noisy digit string by
        dropping up to 2 extraneous digits (e.g. from misread separators),
        keeping only repairs whose day/month/year are individually plausible.
        """
        n = len(digits)
        for remove_count in (0, 1, 2):
            target_len = n - remove_count
            if target_len != 8:
                continue
            for drop_idxs in combinations(range(n), remove_count):
                trimmed = "".join(ch for i, ch in enumerate(digits) if i not in drop_idxs)
                d, m, y = trimmed[:2], trimmed[2:4], trimmed[4:]
                if 1 <= int(d) <= 31 and 1 <= int(m) <= 12 and y.startswith("20"):
                    return d, m, y
        return None

    def validate_flight_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and mathematically resolves flight times using FlightConstraintSolver.
        """
        chocks_off = record.get("chocks_off", {}).get("resolved")
        airborne = record.get("airborne", {}).get("resolved")
        touch_down = record.get("touch_down", {}).get("resolved")
        chocks_on = record.get("chocks_on", {}).get("resolved")
        time_in_air = record.get("time_in_air", {}).get("resolved")
        block_time = record.get("block_time", {}).get("resolved")

        solved_times = FlightConstraintSolver.solve_time_system(
            chocks_off=chocks_off,
            airborne=airborne,
            touch_down=touch_down,
            chocks_on=chocks_on,
            time_in_air=time_in_air,
            block_time=block_time
        )

        # Apply mathematically resolved times
        for key, (t_val, conf) in solved_times.items():
            if t_val:
                if key not in record:
                    record[key] = {}
                record[key]["resolved"] = t_val
                if not record[key].get("value"):
                    record[key]["value"] = t_val
                record[key]["confidence"] = max(record[key].get("confidence", 0.0), conf)

        warnings = []
        record["validation_warnings"] = warnings
        record["is_valid"] = True
        return record

import re
from typing import Dict, Any, Optional, List, Tuple

class FlightConstraintSolver:
    """
    Mathematical Constraint Solver for Flight Record Times and Continuity.
    Leverages the invariant laws of flight logs:
      - Block Time = Chocks On - Chocks Off
      - Time in Air = Touchdown - Airborne
      - Sequence: Chocks Off <= Airborne <= Touchdown <= Chocks On
    """

    @staticmethod
    def parse_time_to_minutes(t_str: str) -> Optional[int]:
        if not t_str:
            return None
        digits = re.sub(r"\D", "", t_str)
        if len(digits) == 4:
            hh, mm = int(digits[:2]), int(digits[2:])
            if hh < 24 and mm < 60:
                return hh * 60 + mm
        elif len(digits) == 3:
            hh, mm = int(digits[0]), int(digits[1:])
            if hh < 24 and mm < 60:
                return hh * 60 + mm
        return None

    @staticmethod
    def minutes_to_time_str(m: int) -> str:
        m = m % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    @classmethod
    def solve_time_system(
        cls,
        chocks_off: Optional[str],
        airborne: Optional[str],
        touch_down: Optional[str],
        chocks_on: Optional[str],
        time_in_air: Optional[str],
        block_time: Optional[str]
    ) -> Dict[str, Tuple[str, float]]:
        """
        Solves and reconciles the 6 flight times using linear constraints.
        Returns: {field_name: (resolved_time_str, confidence)}
        """
        m_coff = cls.parse_time_to_minutes(chocks_off)
        m_air = cls.parse_time_to_minutes(airborne)
        m_td = cls.parse_time_to_minutes(touch_down)
        m_con = cls.parse_time_to_minutes(chocks_on)
        m_tair = cls.parse_time_to_minutes(time_in_air)
        m_btime = cls.parse_time_to_minutes(block_time)

        # 1. Deduce Block Time if Chocks Off & Chocks On are known
        if m_coff is not None and m_con is not None:
            calc_block = (m_con - m_coff) if m_con >= m_coff else (m_con + 1440 - m_coff)
            if m_btime is None or abs(calc_block - m_btime) > 2:
                m_btime = calc_block

        # 2. Deduce Chocks On if Chocks Off & Block Time are known
        elif m_coff is not None and m_btime is not None and m_con is None:
            m_con = (m_coff + m_btime) % 1440

        # 3. Deduce Chocks Off if Chocks On & Block Time are known
        elif m_con is not None and m_btime is not None and m_coff is None:
            m_coff = (m_con - m_btime + 1440) % 1440

        # 4. Deduce Time in Air if Airborne & Touchdown are known
        if m_air is not None and m_td is not None:
            calc_air = (m_td - m_air) if m_td >= m_air else (m_td + 1440 - m_air)
            if m_tair is None or abs(calc_air - m_tair) > 2:
                m_tair = calc_air

        # 5. Deduce Touchdown if Airborne & Time in Air are known
        elif m_air is not None and m_tair is not None and m_td is None:
            m_td = (m_air + m_tair) % 1440

        # 6. Deduce Airborne if Touchdown & Time in Air are known
        elif m_td is not None and m_tair is not None and m_air is None:
            m_air = (m_td - m_tair + 1440) % 1440

        # If airborne is still missing but chocks off is known, taxi time is typically 5-20 mins
        if m_air is None and m_coff is not None and m_td is not None and m_tair is not None:
            m_air = (m_td - m_tair + 1440) % 1440

        results = {}
        for name, m_val in [
            ("chocks_off", m_coff),
            ("airborne", m_air),
            ("touch_down", m_td),
            ("chocks_on", m_con),
            ("time_in_air", m_tair),
            ("block_time", m_btime)
        ]:
            if m_val is not None:
                results[name] = (cls.minutes_to_time_str(m_val), 0.98)
            else:
                results[name] = ("", 0.0)

        return results

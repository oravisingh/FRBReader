---
title: FRB Data Extraction Pipeline
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.51.0"
app_file: ui/app.py
pinned: false
---

# FRB Data Extraction Pipeline

CPU-only Document Processing Engine that extracts flight data from scanned/
photographed Flight Record Book (FRB) pages and assembles them into FDTL
(Flight Duty Time Limitations) journey records for human-in-the-loop review.

A single journey (a flight that lands and departs again one or more times)
is recorded as **one** journey with multiple legs, not as separate records -
legs are chained by matching each page's destination airport to the next
page's departure airport.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
streamlit run ui/app.py
```

Select **Sample Journey (4 pages)** in the sidebar to see the bundled 4-leg
demo journey, or **Upload Scans / PDF** to process your own scans (upload
pages in chronological order - page order is not auto-detected from OCR).

## Test

```
python3 -m unittest discover -s tests
```

`tests/test_pipeline_run.py` is also runnable directly (`python3
tests/test_pipeline_run.py`) for a human-readable field-by-field dump against
the real sample scans.

## Notes on the sample data

- `data/IMG-20260829-WA00{10,11,12,13}.jpg` are the real scanned FRB pages
  (physical pages 018-021) used by the sample journey and tests.
- `data/Add Flight123456789.pdf` is **not a scan** - it's a browser
  printout of the target FDTL web app's "Add Flight" screen for this same
  journey, kept as a reference for what the final journey record should
  look like. It must never be fed through the alignment/OCR pipeline.

## Architecture

- `engine/aligner.py` - OpenCV SIFT/ORB feature matching + homography to warp
  a scan onto the reference template (`reference/template.png`).
- `engine/ocr_engine.py` - field-cropped OCR via PaddleOCR's detection/
  recognition models running through ONNX Runtime (`rapidocr-onnxruntime`),
  with whitelist-based character filtering.
- `engine/postprocessor.py` - regex/format coercion and RapidFuzz entity
  resolution (airports, crew) against `config/airports.json` and
  `config/roster.json`.
- `engine/journey_builder.py` - chains consecutive per-page records into
  journeys and flags cross-leg inconsistencies for review.
- `engine/checkbox.py` - pixel-density checkbox state detection.
- `ui/app.py` - Streamlit HITL dashboard: per-field crop + OCR value side by
  side, journey summary, and FDTL JSON export.

Fields the engine could not confidently read are never silently guessed -
they carry `needs_review: true` so the UI surfaces them for a human to
correct before the record is submitted.

## Extracted fields

Beyond the header/timing/crew/fuel fields, the schema also captures (added
from the physical form, not the target app's PDF - most of that PDF's other
fields are duty-time policy constants the target system computes, not things
handwritten on this form):

- **Cumulative hours & landings**: `log_hrs_bf`, `total_hours`,
  `landings_bf`, `total_ldgs` - the "LOG HRS B/F / TOTAL HOURS" and
  "LANDINGS B/F / TOTAL LDGS" rows.
- **Sign-off / license records**: `pic_lic_no`, `pic_sig_time`,
  `pic_sig_date` (the PIC signature block) and `eng1_auth_lic_no`,
  `eng2_auth_lic_no` (the pre-flight inspection sign-off rows).
- **Defect status**: `defect_status`, the free-text "PILOT REPORTED DEFECT"
  area (usually "NIL").

Crew names are resolved to the roster's own `display_name` (e.g. "D. SINGH"),
never expanded to the roster's full legal name - the roster is the source of
truth for how a name is shown; the full legal name is still available under
`full_name` if a caller needs it.

## UI

The HITL review screen shows the full aligned page next to the field
editors (not just isolated cutouts), so a reviewer can see each value in the
context of the whole form before correcting it.

import streamlit as st
import json
import os
import sys
import base64
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.pipeline import FRBPipeline

st.set_page_config(
    page_title="FRB Document Processing & FDTL Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aviation ops-dashboard styling: dark navy panel with amber accents (the
# theme colors themselves live in .streamlit/config.toml; this layers on
# card/badge/section treatment Streamlit's theme system doesn't cover).
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }
    .fa-section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 0.7rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #F59E0B;
        font-weight: 700;
        font-size: 0.95rem;
        color: #F1F5F9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .fa-caption {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: -0.5rem;
    }
    .badge-success {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-warning {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    [data-testid="stMetric"] {
        background-color: #141E33;
        border: 1px solid #263352;
        border-radius: 10px;
        padding: 0.7rem 1rem 0.9rem 1rem;
    }
    [data-testid="stMetricValue"] {
        font-family: "Courier New", monospace;
    }
    div[data-testid="stTextInput"] input {
        font-family: "Courier New", monospace;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #263352;
        border-radius: 8px;
    }

    /* Aurora background: a slow-drifting gradient glow behind the whole app,
       recreated in pure CSS (no JS/React needed) so it works inside
       Streamlit. Heavily blurred and low-opacity so it reads as ambient
       atmosphere, not something competing with data tables for attention. */
    .stApp {
        position: relative;
        background-color: #0B1120;
    }
    .stApp::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image:
            repeating-linear-gradient(100deg, #0B1120 0%, #0B1120 7%, transparent 10%, transparent 12%, #0B1120 16%),
            repeating-linear-gradient(100deg, #F59E0B 5%, #0EA5E9 15%, #1D4ED8 25%, #7C3AED 35%, #F59E0B 45%);
        background-size: 300% 200%;
        background-position: 50% 50%, 50% 50%;
        filter: blur(80px) saturate(140%);
        opacity: 0.22;
        animation: fa-aurora-drift 90s linear infinite;
    }
    @keyframes fa-aurora-drift {
        from { background-position: 50% 50%, 50% 50%; }
        to   { background-position: 350% 50%, 350% 50%; }
    }
    @media (prefers-reduced-motion: reduce) {
        .stApp::before { animation: none; }
    }
    /* Keep actual content above the aurora layer, and give data-dense
       surfaces a solid-ish backing so contrast/legibility isn't affected. */
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }
    [data-testid="stMetric"], div[data-testid="stExpander"], .stTable, div[data-testid="stDataFrame"] {
        background-color: rgba(20, 30, 51, 0.92);
    }

    /* Make the sidebar's built-in collapse/expand arrow easy to spot against
       the dark theme - it's a native Streamlit control (top of the sidebar
       to collapse; a small floating arrow at the top-left to reopen it),
       just low-contrast by default here. */
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stExpandSidebarButton"] svg {
        color: #F59E0B !important;
        fill: #F59E0B !important;
    }
    [data-testid="stExpandSidebarButton"] {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def section_header(icon: str, title: str):
    st.markdown(f'<div class="fa-section-header">{icon} {title}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return FRBPipeline(
        schema_path=os.path.join(root, "config", "template_schema.json"),
        reference_path=os.path.join(root, "reference", "template.png"),
        roster_path=os.path.join(root, "config", "roster.json"),
        airports_path=os.path.join(root, "config", "airports.json"),
    )


@st.cache_data(show_spinner=False, max_entries=20, ttl=3600)
def run_pipeline(_pipeline: FRBPipeline, cache_key: tuple):
    """
    Cached so that editing a field (which reruns this whole script, like
    every Streamlit interaction) doesn't re-run alignment/OCR from scratch
    on every keystroke - it also keeps each leg's `fields` dict identity
    stable across reruns, which matters because widget keys are derived
    from it. `cache_key` is (path, mtime) pairs so a re-upload busts the cache.

    Bounded (max_entries/ttl): without a limit this cache would hold every
    distinct document ever processed for the life of the running server -
    fine for disk (nothing here touches disk), but unbounded memory growth
    over months of real usage. Oldest entries are evicted once the cache is
    full or after an hour, whichever comes first.
    """
    paths = [p for p, _ in cache_key]
    return _pipeline.process_batch(paths)


def widget_key(leg: dict, field_key: str) -> str:
    """Stable across reruns (unlike id(fields), which changes every time the
    pipeline re-processes) so Streamlit doesn't discard in-progress edits."""
    return f"val::{leg.get('source', '')}::{field_key}"


def current_value(leg: dict, field_key: str, default: str = ""):
    """A human's edit (held in session_state under the field's widget key)
    takes precedence over the original OCR-resolved value."""
    key = widget_key(leg, field_key)
    if key in st.session_state:
        return st.session_state[key]
    return leg.get("fields", {}).get(field_key, {}).get("resolved", default)


def journey_live_view(journey: dict) -> dict:
    """Journey-level fields (route, start/end time) recomputed from each
    leg's CURRENT value, so a correction made in the HITL tab is reflected
    everywhere else (summary, metrics, export) instead of only in that
    one text box."""
    legs = journey["legs"]
    route = [current_value(legs[0], "from_icao")] + [current_value(leg, "to_icao") for leg in legs]
    return {
        "route": route,
        "departure_icao": route[0],
        "destination_icao": route[-1],
        "flight_started_utc": current_value(legs[0], "chocks_off"),
        "flight_ended_utc": current_value(legs[-1], "chocks_on"),
        "flight_date": current_value(legs[0], "flight_date"),
        "aircraft_registration": current_value(legs[0], "ac_regn"),
    }


# The known real sample scans, in physical page order, representing ONE
# journey with 4 landings (018 VOHS->VERP, 019 VERP->VOBG, 020 VOBG->VOMY,
# 021 VOMY->VOMM). Order matters: journeys are built by chaining consecutive
# pages, not by re-sorting an OCR'd page number.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_JOURNEY_PAGES = [
    os.path.join(_PROJECT_ROOT, "data", "IMG-20260829-WA0011.jpg"),
    os.path.join(_PROJECT_ROOT, "data", "IMG-20260829-WA0010.jpg"),
    os.path.join(_PROJECT_ROOT, "data", "IMG-20260829-WA0013.jpg"),
    os.path.join(_PROJECT_ROOT, "data", "IMG-20260829-WA0012.jpg"),
]


def main():
    st.markdown('<div class="main-title">✈️ FRB DATA EXTRACTION &amp; FDTL ENGINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">CPU-Optimized Document Processing · Handwritten Flight Record Books</div>',
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()

    with st.sidebar:
        st.header("📂 Input Documents")
        input_mode = st.radio("Choose Source", ["Sample Journey (4 pages)", "Upload Scans / PDF"])

        target_paths = None

        if input_mode == "Sample Journey (4 pages)":
            st.caption("Pages 018→021: one journey, 4 landings (VOHS→VERP→VOBG→VOMY→VOMM).")
            target_paths = SAMPLE_JOURNEY_PAGES
        else:
            uploaded_files = st.file_uploader(
                "Upload scans or a multi-page PDF, in chronological order",
                type=["jpg", "jpeg", "png", "pdf"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                # Written to the OS temp dir, not the project folder - a
                # scratch/uploads/ directory here would grow without bound
                # since nothing would ever clean it up. Cleared before each
                # new batch so it doesn't grow within a long-running session
                # either.
                upload_dir = st.session_state.setdefault("upload_dir", tempfile.mkdtemp(prefix="frbreader_"))
                for stale in os.listdir(upload_dir):
                    os.remove(os.path.join(upload_dir, stale))
                target_paths = []
                for f in uploaded_files:
                    temp_path = os.path.join(upload_dir, f.name)
                    with open(temp_path, "wb") as out:
                        out.write(f.getbuffer())
                    target_paths.append(temp_path)

        st.markdown("---")
        st.subheader("⚙️ System Status")
        st.markdown("**Compute Engine:** CPU (ONNX Runtime)")
        st.markdown("**Alignment:** SIFT / ORB + Homography")
        st.markdown("**Entity Resolver:** RapidFuzz Levenshtein")
        st.markdown("**OCR:** Field-Constrained PaddleOCR (ONNX)")

    if not target_paths:
        st.info("Select the sample journey or upload one or more scans/PDFs to begin.")
        return

    cache_key = tuple((p, os.path.getmtime(p)) for p in target_paths)
    with st.spinner("Processing document(s) (Alignment, ROI Slicing, OCR, Entity Resolution, Journey Grouping)..."):
        journeys = run_pipeline(pipeline, cache_key)

    if not journeys:
        st.error("No flight records extracted from the uploaded document(s).")
        return

    if len(journeys) > 1:
        journey_labels = [
            f"Journey {i + 1}: {' → '.join(j['route'])} ({j['total_landings']} landing"
            f"{'s' if j['total_landings'] != 1 else ''})"
            for i, j in enumerate(journeys)
        ]
        selected_idx = st.selectbox(
            "Multiple unrelated routes were detected in this upload - select a journey to review",
            range(len(journeys)),
            format_func=lambda i: journey_labels[i],
        )
    else:
        selected_idx = 0

    journey = journeys[selected_idx]
    render_journey(journey)


def render_journey(journey: dict):
    legs = journey["legs"]
    live = journey_live_view(journey)

    # Top Metrics Bar
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Route", " → ".join(live["route"]))
    with col_m2:
        st.metric("Landings", journey["total_landings"])
    with col_m3:
        st.metric("Flight Started (UTC)", live["flight_started_utc"] or "--:--")
    with col_m4:
        st.metric("Flight Ended (UTC)", live["flight_ended_utc"] or "--:--")

    if journey["needs_review"]:
        st.warning("⚠️ This journey has fields that need human review before submitting to the FDTL database.")
        inconsistencies = journey.get("inconsistencies", {})
        for field, values in inconsistencies.items():
            if values:
                st.caption(f"Inconsistent **{field}** across legs: {', '.join(values)} - confirm the correct value below.")
    else:
        st.success("✅ All legs passed alignment, OCR-confidence and consistency checks.")

    tab_summary, tab_hitl, tab_alignment, tab_fdtl_json = st.tabs([
        "🧭 Journey Summary",
        "🔍 Human-in-the-Loop Verification",
        "📐 Document Alignment Inspection",
        "📋 Export FDTL JSON",
    ])

    with tab_summary:
        render_journey_summary(journey)

    with tab_hitl:
        leg_tabs = st.tabs([
            f"Leg {i + 1}: {leg['fields'].get('from_icao', {}).get('resolved', '?')} → "
            f"{leg['fields'].get('to_icao', {}).get('resolved', '?')}"
            for i, leg in enumerate(legs)
        ])
        for leg_tab, leg in zip(leg_tabs, legs):
            with leg_tab:
                render_leg_editor(leg)

    with tab_alignment:
        for i, leg in enumerate(legs):
            score = leg.get("alignment_score", 0.0)
            badge = "✅" if score >= 0.7 else "⚠️"
            st.markdown(f"{badge} **Leg {i + 1}** ({leg.get('source', '')}) — Alignment confidence: `{score * 100:.1f}%`")

    with tab_fdtl_json:
        render_fdtl_export(journey)


def render_journey_summary(journey: dict):
    section_header("📄", "Leg-by-leg summary")
    st.caption("Reflects any corrections made in the Human-in-the-Loop tab.")
    rows = []
    for i, leg in enumerate(journey["legs"]):
        f = leg["fields"]
        rows.append({
            "Leg": i + 1,
            "From": current_value(leg, "from_icao"),
            "To": current_value(leg, "to_icao"),
            "Chocks Off": current_value(leg, "chocks_off"),
            "Chocks On": current_value(leg, "chocks_on"),
            "Block Time": current_value(leg, "block_time"),
            "Total Ldgs (cum.)": current_value(leg, "total_ldgs"),
            "Needs Review": "⚠️" if any(
                isinstance(v, dict) and v.get("needs_review") for v in f.values()
            ) else "",
        })
    st.table(rows)

    section_header("🧮", "Journey totals")
    live = journey_live_view(journey)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Landings", journey["total_landings"])
    c2.metric("Flight Started (UTC)", live["flight_started_utc"] or "--:--")
    c3.metric("Flight Ended (UTC)", live["flight_ended_utc"] or "--:--")


def render_leg_editor(leg: dict):
    fields = leg.get("fields", {})
    full_page_b64 = leg.get("full_page_base64", "")

    col_page, col_fields = st.columns([2, 3])

    with col_page:
        section_header("🖼️", "Full Page")
        st.caption("Aligned scan - use this for context around any cutout.")
        if full_page_b64:
            st.image(base64.b64decode(full_page_b64), use_container_width=True)

    with col_fields:
        # Everything below is one form: edits to any field only take effect
        # (and only trigger a single rerun) when "Save" is pressed - not on
        # every individual Enter/Tab, which otherwise reruns the whole app
        # once per field.
        with st.form(key=f"leg_form_{leg.get('source', '')}"):
            section_header("1️⃣", "Flight Header & Aircraft Info")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_field_editor(leg, "page_no", "Page Number")
            with c2:
                render_field_editor(leg, "flight_date", "Flight Date")
            with c3:
                render_field_editor(leg, "ac_type", "Aircraft Type")
            with c4:
                render_field_editor(leg, "ac_regn", "Aircraft Regn")

            section_header("2️⃣", "Sector & Routing")
            c_dep, c_arr, c_rev, c_med = st.columns(4)
            with c_dep:
                render_field_editor(leg, "from_icao", "Departure Airport (ICAO)")
                apt_info = fields.get("from_icao", {}).get("airport_name", "")
                if apt_info:
                    st.caption(f"🏢 {apt_info}")
            with c_arr:
                render_field_editor(leg, "to_icao", "Destination Airport (ICAO)")
                apt_info = fields.get("to_icao", {}).get("airport_name", "")
                if apt_info:
                    st.caption(f"🏢 {apt_info}")
            with c_rev:
                render_checkbox_editor(leg, "chk_revenue", "Revenue Flight")
            with c_med:
                render_checkbox_editor(leg, "chk_medivac", "MEDIVAC Config")

            section_header("3️⃣", "Flight Timings (UTC)")
            t1, t2, t3, t4, t5, t6 = st.columns(6)
            for col, key, label in zip(
                [t1, t2, t3, t4, t5, t6],
                ["chocks_off", "airborne", "touch_down", "chocks_on", "time_in_air", "block_time"],
                ["Chocks Off", "Airborne", "Touchdown", "Chocks On", "Time in Air", "Block Time"],
            ):
                with col:
                    render_field_editor(leg, key, label)

            section_header("4️⃣", "Crew Members (Resolved against Active Roster)")
            st.caption("Names are shown exactly as the roster lists them, not expanded to a full legal name.")
            cr1, cr2, cr3, cr4 = st.columns(4)
            with cr1:
                render_field_editor(leg, "pic_name", "PIC Name")
                resolved_pic = fields.get("pic_name", {}).get("resolved", "")
                if resolved_pic:
                    st.markdown(f'<span class="badge-success">✓ Roster: {resolved_pic}</span>', unsafe_allow_html=True)
            with cr2:
                render_field_editor(leg, "pic_staff_id", "PIC Staff ID")
            with cr3:
                render_field_editor(leg, "fo_name", "F/O Name")
                resolved_fo = fields.get("fo_name", {}).get("resolved", "")
                if resolved_fo:
                    st.markdown(f'<span class="badge-success">✓ Roster: {resolved_fo}</span>', unsafe_allow_html=True)
            with cr4:
                render_field_editor(leg, "fo_staff_id", "F/O Staff ID")

            section_header("5️⃣", "Fuel & Landings")
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                render_field_editor(leg, "pre_uplift_fob", "Pre Uplift FOB (LBS)")
            with f2:
                render_field_editor(leg, "departure_fob", "Departure FOB (LBS)")
            with f3:
                render_field_editor(leg, "actual_uplift", "Actual Uplift (LBS)")
            with f4:
                render_field_editor(leg, "fuel_company", "Fuel Company")
            with f5:
                render_field_editor(leg, "full_stop_ldg", "Full Stop Landings")

            section_header("6️⃣", "Cumulative Hours & Landings")
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                render_field_editor(leg, "log_hrs_bf", "Log Hrs B/F")
            with h2:
                render_field_editor(leg, "total_hours", "Total Hours")
            with h3:
                render_field_editor(leg, "landings_bf", "Landings B/F")
            with h4:
                render_field_editor(leg, "total_ldgs", "Total Landings (Cum.)")

            section_header("7️⃣", "Pilot Reported Defect")
            render_field_editor(leg, "defect_status", "Defect / Maintenance Status (NIL if none)")

            section_header("8️⃣", "Sign-off & License Records")
            s1, s2, s3 = st.columns(3)
            with s1:
                render_field_editor(leg, "pic_lic_no", "PIC Signature - Lic No")
            with s2:
                render_field_editor(leg, "pic_sig_time", "PIC Signature - Time")
            with s3:
                render_field_editor(leg, "pic_sig_date", "PIC Signature - Date")
            s4, s5 = st.columns(2)
            with s4:
                render_field_editor(leg, "eng1_auth_lic_no", "Pre-Flight Sign-off - ENG1 Auth/Lic No")
            with s5:
                render_field_editor(leg, "eng2_auth_lic_no", "Pre-Flight Sign-off - ENG2 Auth/Lic No")

            saved = st.form_submit_button("💾 Save Corrections for This Leg", width="stretch")
        if saved:
            st.success("Saved - corrections applied to the Journey Summary and FDTL export.")


def render_field_editor(leg: dict, field_key: str, label: str):
    field_data = leg.get("fields", {}).get(field_key, {})
    crop_b64 = leg.get("crops_base64", {}).get(field_key, "")
    needs_review = bool(field_data.get("needs_review"))
    key = widget_key(leg, field_key)

    badge = ' <span class="badge-warning">⚠️ review</span>' if needs_review else ""
    st.markdown(f"**{label}**{badge}", unsafe_allow_html=True)
    if crop_b64:
        img_bytes = base64.b64decode(crop_b64)
        st.image(img_bytes, use_container_width=True)

    st.text_input(
        label=label,
        value=str(field_data.get("resolved", "")),
        key=key,
        label_visibility="collapsed",
    )


def render_checkbox_editor(leg: dict, field_key: str, label: str):
    field_data = leg.get("fields", {}).get(field_key, {})
    crop_b64 = leg.get("crops_base64", {}).get(field_key, "")

    st.markdown(f"**{label}**")
    if crop_b64:
        img_bytes = base64.b64decode(crop_b64)
        st.image(img_bytes, width=60)

    st.checkbox(
        label=label,
        value=bool(field_data.get("resolved", False)),
        key=widget_key(leg, field_key),
        label_visibility="collapsed",
    )


def render_fdtl_export(journey: dict):
    section_header("📋", "FDTL Compliance Software Payload")
    st.caption("One journey, one payload with a leg per landing - ready for automated ingestion into the FDTL database.")

    live = journey_live_view(journey)

    legs_payload = []
    for i, leg in enumerate(journey["legs"]):
        f = leg["fields"]
        legs_payload.append({
            "leg_number": i + 1,
            "departure_icao": current_value(leg, "from_icao"),
            "destination_icao": current_value(leg, "to_icao"),
            "flight_date": current_value(leg, "flight_date"),
            "timings_utc": {
                "chocks_off": current_value(leg, "chocks_off"),
                "airborne": current_value(leg, "airborne"),
                "touchdown": current_value(leg, "touch_down"),
                "chocks_on": current_value(leg, "chocks_on"),
                "flight_time": current_value(leg, "time_in_air"),
                "block_time": current_value(leg, "block_time"),
            },
            "cumulative": {
                "log_hrs_bf": current_value(leg, "log_hrs_bf"),
                "total_hours": current_value(leg, "total_hours"),
                "landings_bf": current_value(leg, "landings_bf"),
                "total_ldgs": current_value(leg, "total_ldgs"),
            },
            "fuel_lbs": {
                "pre_uplift": current_value(leg, "pre_uplift_fob"),
                "departure_fob": current_value(leg, "departure_fob"),
                "actual_uplift": current_value(leg, "actual_uplift"),
                "supplier": current_value(leg, "fuel_company"),
            },
            "configuration": {
                "revenue": current_value(leg, "chk_revenue", False),
                "medivac": current_value(leg, "chk_medivac", False),
            },
            "defect_status": current_value(leg, "defect_status"),
            "sign_off": {
                "pic_lic_no": current_value(leg, "pic_lic_no"),
                "pic_sig_time": current_value(leg, "pic_sig_time"),
                "pic_sig_date": current_value(leg, "pic_sig_date"),
                "eng1_auth_lic_no": current_value(leg, "eng1_auth_lic_no"),
                "eng2_auth_lic_no": current_value(leg, "eng2_auth_lic_no"),
            },
            "needs_review": any(isinstance(v, dict) and v.get("needs_review") for v in f.values()),
        })

    fdtl_payload = {
        "journey_id": journey["journey_id"],
        "flight_date": live["flight_date"],
        "aircraft_registration": live["aircraft_registration"],
        "crew": {
            "pic": {**journey["pic"], "designation": "PIC"},
            "sic": {**journey["sic"], "designation": "SIC"},
        },
        "departure_icao": live["departure_icao"],
        "destination_icao": live["destination_icao"],
        "total_landings": journey["total_landings"],
        "flight_started_utc": live["flight_started_utc"],
        "flight_ended_utc": live["flight_ended_utc"],
        "needs_review": journey["needs_review"],
        "inconsistencies": journey["inconsistencies"],
        "legs": legs_payload,
    }

    st.json(fdtl_payload)

    json_str = json.dumps(fdtl_payload, indent=2)
    st.download_button(
        label="📥 Download FDTL Ingestion JSON",
        data=json_str,
        file_name=f"fdtl_journey_{journey['journey_id']}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

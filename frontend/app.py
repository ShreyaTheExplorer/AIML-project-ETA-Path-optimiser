"""Streamlit UI for Delivery ETA Predictor and Route Optimizer."""

from datetime import date, time, datetime, timedelta
import copy
import os
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = lambda: True

from optimizer.local_optima_demo import run_local_optima_demo
from data.loader import (
    load_cities_data,
    load_all_gazetteers,
    get_city_metadata,
    get_city_gazetteer,
    resolve_context,
)

# ---------------------------------------------------------------------------
# Timezone & Configuration
# ---------------------------------------------------------------------------

IST_ZONE = ZoneInfo("Asia/Kolkata")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------------------
# Central Application Defaults (DEFAULTS Dict)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "city_key": "bengaluru",
    "start_location": "Indiranagar",
    "stops": [
        "MG Road",
        "Koramangala",
        "Jayanagar",
        "HSR Layout",
        "BTM Layout",
        "Marathahalli",
        "Bellandur",
    ],
    "vehicle_type": "Motorcycle",
    "vehicle_condition": 2,
    "rider_age": 25,
    "rider_rating": 4.5,
    "festival": "No",
    "city_type": "Metropolitan",
    "algorithm": "held_karp",
}

# Indian city landmarks matching the training domain (kept for backward compatibility)
LANDMARKS = [
    {"id": "Koregaon Park",       "lat": 18.5362, "lon": 73.8938},
    {"id": "Shivajinagar",        "lat": 18.5204, "lon": 73.8567},
    {"id": "Kothrud",             "lat": 18.5074, "lon": 73.8077},
    {"id": "Aundh",               "lat": 18.5642, "lon": 73.7769},
    {"id": "Baner",               "lat": 18.5590, "lon": 73.7868},
    {"id": "Viman Nagar",         "lat": 18.5679, "lon": 73.9143},
    {"id": "Magarpatta",          "lat": 18.5089, "lon": 73.9260},
    {"id": "Hadapsar",            "lat": 18.5018, "lon": 73.9260},
    {"id": "Pimpri",              "lat": 18.6208, "lon": 73.7986},
    {"id": "Chinchwad",           "lat": 18.6279, "lon": 73.7956},
    {"id": "Wakad",               "lat": 18.5975, "lon": 73.7617},
    {"id": "Hinjewadi",           "lat": 18.5912, "lon": 73.7389},
    {"id": "Sinhagad Road",       "lat": 18.4667, "lon": 73.8167},
    {"id": "Katraj",              "lat": 18.4529, "lon": 73.8616},
    {"id": "Pune Station",        "lat": 18.5286, "lon": 73.8741},
]


def load_all_localities_gazetteer() -> dict[str, dict]:
    """Load combined gazetteer of all localities across cities (backward compatible)."""
    pcmc_names = {
        "Hinjewadi", "Chinchwad", "Wakad", "Pimpri", "Akurdi", "Nigdi",
        "Ravet", "Bhosari", "Moshi", "Pimple Saudagar", "Pimple Gurav",
        "Pimple Nilakh", "Sangvi", "Dapodi", "Kasarwadi"
    }
    try:
        all_gaz = load_all_gazetteers()
        combined = {}
        for ckey, locs in all_gaz.items():
            for name, meta in locs.items():
                city_str = "PCMC" if (ckey == "pune" and name in pcmc_names) else str(meta["city_key"]).capitalize()
                combined[name] = {
                    "lat": float(meta["lat"]),
                    "lon": float(meta["lon"]),
                    "city": city_str,
                    "city_key": meta["city_key"],
                }
        if combined:
            return combined
    except Exception:
        pass
    return {
        lm["id"]: {"lat": lm["lat"], "lon": lm["lon"], "city": "Pune", "city_key": "pune"}
        for lm in LANDMARKS
    }


GAZETTEER = load_all_localities_gazetteer()

# Schema-valid options (matching api/schemas.py)
WEATHER_OPTIONS   = ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "Unknown"]
TRAFFIC_OPTIONS   = ["Low", "Medium", "High", "Jam", "Unknown"]
VEHICLE_OPTIONS   = ["Motorcycle", "Scooter", "Bicycle", "Electric Scooter"]
FESTIVAL_OPTIONS  = ["No", "Yes", "Unknown"]
CITY_OPTIONS      = ["Metropolitan", "Urban", "Semi-Urban"]
ALGORITHM_OPTIONS = ["held_karp", "hill_climbing", "exact"]
ALGORITHM_LABELS  = {
    "held_karp":    "Held-Karp (Exact DP)",
    "hill_climbing": "Hill Climbing (Heuristic)",
    "exact":        "Exact (alias for Held-Karp)",
}


def get_default_ist_departure() -> tuple[date, time]:
    """Calculate default departure: today in IST, rounded up to next 30 minutes."""
    now_ist = datetime.now(IST_ZONE)
    if now_ist.minute < 30:
        target_minute = 30
        target_hour = now_ist.hour
        target_date = now_ist.date()
    else:
        target_minute = 0
        if now_ist.hour < 23:
            target_hour = now_ist.hour + 1
            target_date = now_ist.date()
        else:
            target_hour = 0
            target_date = now_ist.date() + timedelta(days=1)
    return target_date, time(target_hour, target_minute)


def build_default_stops_list(stop_ids: list[str], city_key: str = "bengaluru") -> list[dict]:
    out = []
    city_gaz = get_city_gazetteer(city_key)
    for sid in stop_ids:
        if sid in city_gaz:
            loc = city_gaz[sid]
            out.append({
                "id": sid,
                "lat": loc["lat"],
                "lon": loc["lon"],
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            })
    return out


def get_depot_coords(start_name: str, city_key: str | None = None) -> dict[str, float]:
    """Retrieve authoritative coordinates for a starting location from the gazetteer."""
    if city_key:
        try:
            city_gaz = get_city_gazetteer(city_key)
            if start_name in city_gaz:
                return {"lat": float(city_gaz[start_name]["lat"]), "lon": float(city_gaz[start_name]["lon"])}
        except Exception:
            pass
    loc = GAZETTEER.get(start_name)
    if loc is not None:
        return {"lat": float(loc["lat"]), "lon": float(loc["lon"])}
    return {"lat": 12.9716, "lon": 77.5946}


def build_default_applied_config(
    city_key: str = "bengaluru",
    start_loc: str | None = None,
    stops: list[str] | None = None,
) -> dict:
    d_date, d_time = get_default_ist_departure()
    city_meta = get_city_metadata(city_key)
    start_name = start_loc or city_meta["default_start"]
    depot_coords = get_depot_coords(start_name, city_key=city_key)
    dep_iso = f"{d_date.isoformat()}T{d_time.strftime('%H:%M:%S')}"
    ctx = resolve_context(city_key, d_date)

    if stops is not None:
        stop_names = stops
    elif city_key == "bengaluru":
        stop_names = list(DEFAULTS["stops"])
    else:
        stop_names = []

    return {
        "city_key": city_key,
        "city_display": city_meta["display_name"],
        "start_location": start_name,
        "depot_lat": depot_coords["lat"],
        "depot_lon": depot_coords["lon"],
        "departure_date": d_date,
        "departure_time": d_time,
        "departure_iso": dep_iso,
        "rider": {
            "vehicle_type": DEFAULTS["vehicle_type"],
            "vehicle_condition": DEFAULTS["vehicle_condition"],
            "delivery_person_age": float(DEFAULTS["rider_age"]),
            "delivery_person_rating": float(DEFAULTS["rider_rating"]),
        },
        "festival": ctx.get("festival", DEFAULTS["festival"]),
        "city_type": ctx.get("city_type", DEFAULTS["city_type"]),
        "algorithm": DEFAULTS["algorithm"],
        "stops": build_default_stops_list(stop_names, city_key=city_key),
    }


def build_optimize_payload(applied_cfg: dict) -> dict:
    """Build optimize API payload with depot lat/lon derived strictly from gazetteer."""
    city_key = applied_cfg.get("city_key", DEFAULTS["city_key"])
    start_loc = applied_cfg.get("start_location", DEFAULTS["start_location"])
    coords = get_depot_coords(start_loc, city_key=city_key)
    return {
        "city_key": city_key,
        "depot_lat": coords["lat"],
        "depot_lon": coords["lon"],
        "rider": applied_cfg["rider"],
        "festival": applied_cfg["festival"],
        "city_type": applied_cfg["city_type"],
        "stops": applied_cfg["stops"],
        "departure_time": applied_cfg["departure_iso"],
        "algorithm": applied_cfg["algorithm"],
    }


def is_running_in_streamlit() -> bool:
    """Check if code is running inside an active Streamlit runtime or AppTest."""
    return get_script_run_ctx() is not None


# ---------------------------------------------------------------------------
# Map Helper
# ---------------------------------------------------------------------------

def build_map(
    depot: dict,
    stops_coords: dict,
    route_order: list,
    color: str = "#1976D2",
    center: list[float] | None = None,
    zoom_start: int = 12,
) -> folium.Map:
    coords = {"depot": (depot["lat"], depot["lon"]), **stops_coords}
    pts = [coords[n] for n in route_order if n in coords]
    if center is None:
        center = (
            [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
            if pts else [depot["lat"], depot["lon"]]
        )
    m = folium.Map(location=center, zoom_start=zoom_start)
    if len(pts) > 1:
        folium.PolyLine(pts, color=color, weight=4, opacity=0.85).add_to(m)

    depot_html = (
        '<div style="font-size:10px;font-weight:bold;color:white;background:#C62828;'
        'border-radius:4px;padding:2px 8px;border:2px solid white;'
        'box-shadow:1px 1px 3px rgba(0,0,0,.5);">DEPOT</div>'
    )
    folium.Marker(
        [depot["lat"], depot["lon"]],
        tooltip="Depot (Start)",
        icon=folium.DivIcon(html=depot_html),
    ).add_to(m)

    for i, node in enumerate(route_order):
        if node == "depot" or node not in coords:
            continue
        badge = (
            f'<div style="font-size:11px;font-weight:bold;color:white;background:{color};'
            f'border-radius:50%;width:24px;height:24px;display:flex;align-items:center;'
            f'justify-content:center;border:2px solid white;'
            f'box-shadow:1px 1px 3px rgba(0,0,0,.5);">{i}</div>'
        )
        folium.Marker(
            coords[node],
            tooltip=f"{i}. {node}",
            icon=folium.DivIcon(html=badge),
        ).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Streamlit Application Execution
# ---------------------------------------------------------------------------

def run_app():
    # ── State Initialization ──────────────────────────────────────────────
    if "applied_config" not in st.session_state:
        st.session_state["applied_config"] = build_default_applied_config()

    if "stop_configs" not in st.session_state:
        st.session_state["stop_configs"] = {}

    cities_data = load_cities_data()
    city_options = ["bengaluru", "pune", "mumbai", "delhi"]
    city_labels = {k: cities_data[k]["display_name"] for k in city_options}

    # ── On City Change Callback ───────────────────────────────────────────
    def on_city_change():
        chosen_city = st.session_state.get("city_select_key", "bengaluru")
        meta = get_city_metadata(chosen_city)
        def_start = meta["default_start"]
        st.session_state["start_loc_key"] = def_start
        st.session_state["stops_multiselect_key"] = []
        st.session_state["stop_configs"] = {}
        st.session_state.pop("optimize_result", None)
        st.session_state.pop("last_optimized_payload", None)
        st.session_state.pop("last_optimized_config", None)
        st.session_state["applied_config"] = build_default_applied_config(
            city_key=chosen_city,
            start_loc=def_start,
            stops=[],
        )

    # ── Reset Callback ───────────────────────────────────────────────────
    def reset_to_defaults():
        target_date, target_time = get_default_ist_departure()
        st.session_state["city_select_key"] = DEFAULTS["city_key"]
        st.session_state["start_loc_key"] = DEFAULTS["start_location"]
        st.session_state["stops_multiselect_key"] = list(DEFAULTS["stops"])
        st.session_state["dep_date_key"] = target_date
        st.session_state["dep_time_key"] = target_time
        st.session_state["vehicle_type_key"] = DEFAULTS["vehicle_type"]
        st.session_state["vehicle_condition_key"] = DEFAULTS["vehicle_condition"]
        st.session_state["rider_age_key"] = DEFAULTS["rider_age"]
        st.session_state["rider_rating_key"] = float(DEFAULTS["rider_rating"])
        st.session_state["festival_key"] = DEFAULTS["festival"]
        st.session_state["city_type_key"] = DEFAULTS["city_type"]
        st.session_state["algorithm_key"] = DEFAULTS["algorithm"]
        st.session_state["stop_configs"] = {}
        st.session_state["applied_config"] = build_default_applied_config()
        st.session_state.pop("optimize_result", None)
        st.session_state.pop("last_optimized_payload", None)
        st.session_state.pop("last_optimized_config", None)

    # ── Page Header ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Delivery ETA & Route Optimizer",
        page_icon="🚴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚴 Delivery ETA Predictor & Route Optimizer")
    st.caption(
        "Multi-City Route Optimization · XGBoost ETA Predictor · Held-Karp Exact Solver"
    )

    # ── 1. City Selectbox (TOP of sidebar, outside form) ──────────────────
    st.sidebar.subheader("🏙️ City Selection")
    cur_city_key = st.session_state.get("city_select_key", DEFAULTS["city_key"])
    city_idx = city_options.index(cur_city_key) if cur_city_key in city_options else 0

    selected_city = st.sidebar.selectbox(
        "City",
        options=city_options,
        format_func=lambda k: city_labels[k],
        index=city_idx,
        key="city_select_key",
        on_change=on_city_change,
        help="Select the operational city. Localities and coordinates will update accordingly.",
    )

    current_city_meta = get_city_metadata(selected_city)
    current_city_gaz = get_city_gazetteer(selected_city)
    city_locality_names = list(current_city_gaz.keys())

    # ── 2. Starting Location (outside form) ──────────────────────────────
    st.sidebar.subheader("📍 Starting Location (Depot)")
    default_start_val = st.session_state.get("start_loc_key", current_city_meta["default_start"])
    if default_start_val not in city_locality_names:
        default_start_val = current_city_meta["default_start"]
        st.session_state["start_loc_key"] = default_start_val
    start_idx = city_locality_names.index(default_start_val) if default_start_val in city_locality_names else 0

    start_location = st.sidebar.selectbox(
        "Starting location",
        options=city_locality_names,
        index=start_idx,
        key="start_loc_key",
        help=f"Select the depot / origin location (Node 0) in {current_city_meta['display_name']}.",
    )
    depot_info = current_city_gaz[start_location]
    st.sidebar.caption(
        f"Coordinates: `{depot_info['lat']:.4f}° N, {depot_info['lon']:.4f}° E` ({current_city_meta['display_name']})"
    )

    # Automatically keep applied_config start_location and its derived depot coordinates always up to date
    cur_applied = st.session_state.get("applied_config")
    if cur_applied and (cur_applied.get("start_location") != start_location or cur_applied.get("city_key") != selected_city):
        cur_applied["city_key"] = selected_city
        cur_applied["city_display"] = current_city_meta["display_name"]
        cur_applied["start_location"] = start_location
        cur_applied["depot_lat"] = float(depot_info["lat"])
        cur_applied["depot_lon"] = float(depot_info["lon"])

    # ── 3. Select Stops (outside form) ───────────────────────────────────
    st.sidebar.subheader("🗺️ Stops Configuration")
    available_stops = [name for name in city_locality_names if name != start_location]

    current_stops = st.session_state.get(
        "stops_multiselect_key",
        list(DEFAULTS["stops"]) if selected_city == "bengaluru" else []
    )
    # Ensure current_stops only contains available stops from this city
    current_stops = [s for s in current_stops if s in available_stops]
    st.session_state["stops_multiselect_key"] = current_stops

    selected_stops = st.sidebar.multiselect(
        "Select stops",
        options=available_stops,
        default=current_stops,
        key="stops_multiselect_key",
        max_selections=12,
        help=f"Select between 2 and 12 delivery stops in {current_city_meta['display_name']}.",
    )
    st.sidebar.caption(
        "Select 2 to 12 stops. Held-Karp exact solver runs in O(n²·2ⁿ). At least 2 stops required."
    )

    # ── 4. Sample Random Stops Button (outside form) ─────────────────────
    def sample_random_stops_callback():
        avail = [name for name in city_locality_names if name != start_location]
        st.session_state["stops_multiselect_key"] = random.sample(avail, min(4, len(avail)))

    st.sidebar.button(
        "🔀 Sample Random Stops",
        on_click=sample_random_stops_callback,
        help=f"Quickly populate 4 random stops in {current_city_meta['display_name']}.",
    )

    st.sidebar.divider()

    # ── 5. Sidebar Form Wrapping Remaining Controls ─────────────────────
    with st.sidebar.form("config_form"):
        st.subheader("⏰ Departure Time (IST)")
        now_ist = datetime.now(IST_ZONE)
        def_date, def_time = get_default_ist_departure()

        dep_date = st.date_input(
            "Departure Date",
            value=st.session_state.get("dep_date_key", def_date),
            min_value=now_ist.date(),
            key="dep_date_key",
            help="Must be today or in the future in Asia/Kolkata (IST).",
        )
        dep_time = st.time_input(
            "Departure Time",
            value=st.session_state.get("dep_time_key", def_time),
            key="dep_time_key",
            help="Current time rounded up to next 30 minutes in Asia/Kolkata.",
        )

        is_past_time = (dep_date == now_ist.date() and dep_time < now_ist.time())
        if is_past_time:
            st.error(
                f"⚠️ Selected time ({dep_time.strftime('%H:%M')}) is earlier than current IST time ({now_ist.strftime('%H:%M')})."
            )

        st.divider()

        st.subheader("🧑 Rider Details")
        vehicle_type = st.selectbox(
            "Vehicle Type",
            VEHICLE_OPTIONS,
            index=VEHICLE_OPTIONS.index(st.session_state.get("vehicle_type_key", DEFAULTS["vehicle_type"])),
            key="vehicle_type_key",
        )
        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            [0, 1, 2, 3],
            index=st.session_state.get("vehicle_condition_key", DEFAULTS["vehicle_condition"]),
            key="vehicle_condition_key",
            help="0 = poor … 3 = excellent",
        )
        rider_age = st.slider(
            "Rider Age",
            min_value=20, max_value=39,
            value=int(st.session_state.get("rider_age_key", DEFAULTS["rider_age"])),
            key="rider_age_key",
        )
        rider_rating = st.slider(
            "Rider Rating",
            min_value=2.5, max_value=5.0,
            value=float(st.session_state.get("rider_rating_key", DEFAULTS["rider_rating"])),
            step=0.1,
            key="rider_rating_key",
        )

        st.divider()

        st.subheader("⚙️ Route Settings")
        # Pre-resolve default festival based on city and date
        resolved_context = resolve_context(selected_city, dep_date)

        festival = st.selectbox(
            "Festival Day",
            FESTIVAL_OPTIONS,
            index=FESTIVAL_OPTIONS.index(st.session_state.get("festival_key", resolved_context["festival"])),
            key="festival_key",
        )
        city_type = st.selectbox(
            "City Type",
            CITY_OPTIONS,
            index=CITY_OPTIONS.index(st.session_state.get("city_type_key", resolved_context["city_type"])),
            key="city_type_key",
        )
        algorithm = st.selectbox(
            "Optimizer Algorithm",
            ALGORITHM_OPTIONS,
            format_func=lambda k: ALGORITHM_LABELS[k],
            index=ALGORITHM_OPTIONS.index(st.session_state.get("algorithm_key", DEFAULTS["algorithm"])),
            key="algorithm_key",
        )

        st.divider()

        st.subheader("📋 Stop Conditions Editor")
        editor_rows = []
        for stop_id in selected_stops:
            if stop_id in current_city_gaz:
                loc = current_city_gaz[stop_id]
                cfg = st.session_state["stop_configs"].get(stop_id, {})
                editor_rows.append({
                    "id": stop_id,
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "weather": cfg.get("weather", "Sunny"),
                    "traffic": cfg.get("traffic", "Medium"),
                    "multiple_deliveries": int(cfg.get("multiple_deliveries", 0)),
                })

        editor_df = pd.DataFrame(editor_rows)
        if not editor_df.empty:
            edited_table = st.data_editor(
                editor_df,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn("Stop ID", disabled=True),
                    "lat": st.column_config.NumberColumn("Lat", format="%.4f", disabled=True),
                    "lon": st.column_config.NumberColumn("Lon", format="%.4f", disabled=True),
                    "weather": st.column_config.SelectboxColumn("Weather", options=WEATHER_OPTIONS),
                    "traffic": st.column_config.SelectboxColumn("Traffic", options=TRAFFIC_OPTIONS),
                    "multiple_deliveries": st.column_config.NumberColumn("Multi-Del", min_value=0, max_value=3, step=1),
                },
                key="stops_table_editor",
            )
        else:
            edited_table = editor_df
            st.caption("No stops currently selected.")

        st.divider()

        apply_clicked = st.form_submit_button("✅ Apply Filters", type="primary", use_container_width=True)
        reset_clicked = st.form_submit_button("🔄 Reset to Defaults", use_container_width=True, on_click=reset_to_defaults)

    # ── Handle Form Submit ────────────────────────────────────────────────
    if apply_clicked:
        if isinstance(edited_table, pd.DataFrame) and not edited_table.empty:
            for _, r in edited_table.iterrows():
                st.session_state["stop_configs"][r["id"]] = {
                    "weather": str(r["weather"]),
                    "traffic": str(r["traffic"]),
                    "multiple_deliveries": int(r["multiple_deliveries"]),
                }

        now_ist_check = datetime.now(IST_ZONE)
        if dep_date == now_ist_check.date() and dep_time < now_ist_check.time():
            st.sidebar.error("❌ Cannot apply: Departure time cannot be earlier than current IST time.")
        elif len(selected_stops) < 2:
            st.sidebar.error("❌ Cannot apply: At least 2 delivery stops are required.")
        elif len(selected_stops) > 12:
            st.sidebar.error(f"❌ Cannot apply: At most 12 stops allowed (current: {len(selected_stops)}).")
        else:
            stops_payload_list = []
            for stop_id in selected_stops:
                if stop_id in current_city_gaz:
                    loc = current_city_gaz[stop_id]
                    cfg = st.session_state["stop_configs"].get(stop_id, {})
                    stops_payload_list.append({
                        "id": stop_id,
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "weather": cfg.get("weather", "Sunny"),
                        "traffic": cfg.get("traffic", "Medium"),
                        "multiple_deliveries": int(cfg.get("multiple_deliveries", 0)),
                    })

            departure_iso = f"{dep_date.isoformat()}T{dep_time.strftime('%H:%M:%S')}"
            depot_coords = get_depot_coords(start_location, city_key=selected_city)
            st.session_state["applied_config"] = {
                "city_key": selected_city,
                "city_display": current_city_meta["display_name"],
                "start_location": start_location,
                "depot_lat": depot_coords["lat"],
                "depot_lon": depot_coords["lon"],
                "departure_date": dep_date,
                "departure_time": dep_time,
                "departure_iso": departure_iso,
                "rider": {
                    "vehicle_type": vehicle_type,
                    "vehicle_condition": vehicle_condition,
                    "delivery_person_age": float(rider_age),
                    "delivery_person_rating": float(rider_rating),
                },
                "festival": festival,
                "city_type": city_type,
                "algorithm": algorithm,
                "stops": stops_payload_list,
            }
            st.sidebar.success(f"✅ Filters applied for {current_city_meta['display_name']}!")

    # ── Tabs Navigation ──────────────────────────────────────────────────
    tab_route, tab_demo, tab_model = st.tabs(
        ["Route Optimizer", "Local Optima Demo", "Model Comparison"]
    )

    # =======================================================================
    # Tab 1 — Route Optimizer
    # =======================================================================
    with tab_route:
        applied_cfg = st.session_state.get("applied_config", {})

        # Display detected context box
        if applied_cfg:
            city_disp = applied_cfg.get("city_display", current_city_meta["display_name"])
            c_type = applied_cfg.get("city_type", "Metropolitan")
            fest = applied_cfg.get("festival", "No")
            st.info(f"📌 **Detected context: {city_disp} · {c_type} · Festival: {fest}**")

            with st.expander("🔍 Detailed Applied Parameters", expanded=False):
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.markdown(
                    f"**📍 Start (Depot):** {applied_cfg.get('start_location')}<br>"
                    f"<small>{applied_cfg.get('depot_lat', 0.0):.4f}° N, {applied_cfg.get('depot_lon', 0.0):.4f}° E ({city_disp})</small>",
                    unsafe_allow_html=True,
                )
                col_c2.markdown(
                    f"**⏰ Departure (IST):** {applied_cfg.get('departure_iso', '').replace('T', ' ')}<br>"
                    f"<small>City Type: {c_type} · Festival: {fest}</small>",
                    unsafe_allow_html=True,
                )
                col_c3.markdown(
                    f"**🧑 Rider & Solver:** {applied_cfg.get('rider', {}).get('vehicle_type')} · {ALGORITHM_LABELS.get(applied_cfg.get('algorithm', ''), applied_cfg.get('algorithm', ''))}<br>"
                    f"<small>Age: {applied_cfg.get('rider', {}).get('delivery_person_age', 0):.0f} · Rating: {applied_cfg.get('rider', {}).get('delivery_person_rating', 0):.1f}★</small>",
                    unsafe_allow_html=True,
                )

        # Settings changed banner
        last_cfg = st.session_state.get("last_optimized_config")
        result = st.session_state.get("optimize_result")

        if result and last_cfg and last_cfg != applied_cfg:
            st.warning(
                "⚠️ **Settings changed** since the last run. "
                "Results below reflect the previous configuration. "
                "Click **Optimize Route** to recalculate."
            )

        stops_list = applied_cfg.get("stops", [])
        now_ist_now = datetime.now(IST_ZONE)
        is_time_past = (
            applied_cfg.get("departure_date") == now_ist_now.date()
            and applied_cfg.get("departure_time") < now_ist_now.time()
        ) if "departure_date" in applied_cfg else False

        can_optimize = (
            2 <= len(stops_list) <= 12
            and not is_time_past
        )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            clicked = st.button(
                "🚀 Optimize Route",
                type="primary",
                disabled=not can_optimize,
                use_container_width=True,
            )
        with col_info:
            if is_time_past:
                st.error("Departure time in applied filters is earlier than current IST time.")
            elif len(stops_list) < 2:
                st.info(f"At least 2 delivery stops in {applied_cfg.get('city_display', selected_city.title())} must be selected and applied before optimizing.")
            elif not can_optimize:
                st.info("Fix validation issues above before optimizing.")
            else:
                st.success(
                    f"Ready to optimize {len(stops_list)} stop(s) in {applied_cfg.get('city_display', selected_city.title())} · {ALGORITHM_LABELS.get(applied_cfg.get('algorithm', 'held_karp'))}"
                )

        if clicked:
            current_payload = build_optimize_payload(applied_cfg)
            with st.spinner("Running ETA prediction & route optimization…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/optimize",
                        json=current_payload,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.session_state["optimize_result"] = resp.json()
                        st.session_state["last_optimized_payload"] = current_payload
                        st.session_state["last_optimized_config"] = copy.deepcopy(applied_cfg)
                    elif resp.status_code == 422:
                        detail = resp.json().get("detail", resp.text)
                        errs = (
                            [
                                f"• **{' → '.join(str(x) for x in d.get('loc', []) if x != 'body')}**: "
                                f"{d.get('msg', 'Invalid value')}"
                                for d in detail
                            ]
                            if isinstance(detail, list)
                            else [str(detail)]
                        )
                        st.error("Validation Error (422):\n\n" + "\n".join(errs))
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"❌ Cannot reach the API at **{API_URL}**.\n\n"
                        "Start the backend with:\n"
                        "```\npython -m uvicorn api.main:app --reload\n```"
                    )
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out (60 s). Try fewer stops or a simpler algorithm.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Request failed: {exc}")

        # Derive map depot and center strictly from last_optimized_config if available, or applied_cfg
        opt_cfg = st.session_state.get("last_optimized_config", applied_cfg)
        opt_city_key = opt_cfg.get("city_key", selected_city)
        opt_city_meta = get_city_metadata(opt_city_key)
        map_center = [opt_city_meta["center_lat"], opt_city_meta["center_lon"]]
        map_zoom = opt_city_meta.get("map_zoom", 12)
        map_depot_coords = get_depot_coords(
            opt_cfg.get("start_location", opt_city_meta["default_start"]),
            city_key=opt_city_key,
        )

        if result:
            st.divider()

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Algorithm",         ALGORITHM_LABELS.get(result.get("algorithm", ""), result.get("algorithm", "")))
            m2.metric("Baseline Duration", f"{result['baseline_total_minutes']:.1f} min")
            m3.metric("Optimized Duration", f"{result['total_minutes']:.1f} min")

            imp = result.get("improvement_pct", 0)
            delta_str = f"{imp:+.2f}%"
            m4.metric("Improvement", f"{imp:.2f}%", delta=delta_str if imp != 0 else None,
                      delta_color="normal" if imp >= 0 else "inverse")

            # Distance warnings
            dw = result.get("distance_warnings", {})
            if dw.get("count", 0) > 0:
                tr = dw.get("training_range_km", [])
                st.warning(
                    f"⚠️ **{dw['count']} leg(s)** had distances outside the training range "
                    f"({tr[0]}–{tr[1]} km) and were clipped. Predictions for these legs are extrapolations."
                )

            # Route sequences
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"**Baseline Route ({opt_city_meta['display_name']})**")
                st.write(" → ".join(result.get("baseline_route", [])))
            with col_r2:
                st.markdown(f"**Optimized Route ({opt_city_meta['display_name']})**")
                st.write(" → ".join(result.get("route", [])))

            # Maps
            stops_coords = {s["id"]: (s["lat"], s["lon"]) for s in result.get("stops", [])}

            col_before, col_after = st.columns(2)
            with col_before:
                st.subheader(f"Before: Baseline Route ({opt_city_meta['display_name']})")
                m_before = build_map(map_depot_coords, stops_coords, result["baseline_route"], color="#757575", center=map_center, zoom_start=map_zoom)
                st_folium(
                    m_before,
                    width=None, height=420,
                    returned_objects=[],
                    use_container_width=True,
                    key="map_before",
                )
            with col_after:
                st.subheader(f"After: Optimized Route ({opt_city_meta['display_name']})")
                m_after = build_map(map_depot_coords, stops_coords, result["route"], color="#2E7D32", center=map_center, zoom_start=map_zoom)
                st_folium(
                    m_after,
                    width=None, height=420,
                    returned_objects=[],
                    use_container_width=True,
                    key="map_after",
                )

            # Per-stop schedule
            st.subheader(f"📋 Per-Stop Delivery Schedule ({opt_city_meta['display_name']})")
            if result.get("stops"):
                schedule_df = pd.DataFrame(result["stops"])
                rename = {
                    "id":           "Stop ID",
                    "lat":          "Latitude",
                    "lon":          "Longitude",
                    "leg_minutes":  "Leg (min)",
                    "eta_minutes":  "ETA (min)",
                    "arrival_time": "Arrival Time",
                }
                display_cols = [c for c in rename if c in schedule_df.columns]
                schedule_df = schedule_df[display_cols].rename(columns=rename)
                st.dataframe(schedule_df, use_container_width=True, hide_index=True)

            with st.expander("🔍 Full API Response (JSON)"):
                st.json(result)

    # =======================================================================
    # Tab 2 — Local Optima Demo
    # =======================================================================
    with tab_demo:
        st.caption(
            "Hill climbing from random starting routes. Costs come from a synthetic "
            "asymmetric time matrix (seeded), sized to the current number of stops + depot."
        )

        starts = st.slider("Number of random starts", 10, 100, 50, key="demo_starts")
        demo_seed = st.number_input("Random Seed", value=42, step=1, key="demo_seed_val")
        demo_n = max(len(applied_cfg.get("stops", [])) + 1, 4)
        demo   = run_local_optima_demo(num_starts=starts, n_stops=demo_n, seed=int(demo_seed))

        costs = [r["final_cost"] for r in demo]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            range(1, len(costs) + 1),
            costs,
            marker="o",
            linewidth=1,
            markersize=4,
            label="Final hill-climbing cost",
        )
        ax.axhline(min(costs), linestyle="--", color="green",  label=f"Best = {min(costs):.2f} min")
        ax.axhline(max(costs), linestyle="--", color="red",    label=f"Worst = {max(costs):.2f} min")
        ax.set_xlabel("Random starting route")
        ax.set_ylabel("Final route cost (minutes)")
        ax.set_title("Hill Climbing — Effect of Starting Route on Final Cost")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        unique_costs = len(set(round(c, 6) for c in costs))
        st.write(
            f"**{unique_costs}** unique final costs out of {len(costs)} runs → "
            f"hill climbing gets trapped in **{len(costs) - unique_costs}** duplicate local optima."
        )

    # =======================================================================
    # Tab 3 — Model Comparison
    # =======================================================================
    with tab_model:
        md_path = DOCS_DIR / "model_comparison.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            if "## Key Findings & Viva Talking Points" in content:
                content = content.split("## Key Findings & Viva Talking Points")[0].strip()
            st.markdown(content)
        else:
            st.warning(
                "📄 `docs/model_comparison.md` not found. "
                "Run `python models/train.py` to generate it."
            )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if is_running_in_streamlit():
    run_app()

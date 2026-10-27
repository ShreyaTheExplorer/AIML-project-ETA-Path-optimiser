
"""Streamlit UI for Delivery ETA Predictor and Route Optimizer."""

from datetime import date, time, datetime, timedelta
import copy
import os
from pathlib import Path
import random
from zoneinfo import ZoneInfo
import numpy as np
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


# Add project root to Python's import path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
PERFORMANCE_PATH = DOCS_DIR / "model_performance.csv"
COMPARISON_PATH = DOCS_DIR / "model_comparison.md"

# ---------------------------------------------------------------------------
# Central Application Defaults
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

LANDMARKS = [
    {"id": "Koregaon Park", "lat": 18.5362, "lon": 73.8938},
    {"id": "Shivajinagar", "lat": 18.5204, "lon": 73.8567},
    {"id": "Kothrud", "lat": 18.5074, "lon": 73.8077},
    {"id": "Aundh", "lat": 18.5642, "lon": 73.7769},
    {"id": "Baner", "lat": 18.5590, "lon": 73.7868},
    {"id": "Viman Nagar", "lat": 18.5679, "lon": 73.9143},
    {"id": "Magarpatta", "lat": 18.5089, "lon": 73.9260},
    {"id": "Hadapsar", "lat": 18.5018, "lon": 73.9260},
    {"id": "Pimpri", "lat": 18.6208, "lon": 73.7986},
    {"id": "Chinchwad", "lat": 18.6279, "lon": 73.7956},
    {"id": "Wakad", "lat": 18.5975, "lon": 73.7617},
    {"id": "Hinjewadi", "lat": 18.5912, "lon": 73.7389},
    {"id": "Sinhagad Road", "lat": 18.4667, "lon": 73.8167},
    {"id": "Katraj", "lat": 18.4529, "lon": 73.8616},
    {"id": "Pune Station", "lat": 18.5286, "lon": 73.8741},
]


def load_all_localities_gazetteer() -> dict[str, dict]:
    """Load combined gazetteer of all localities across cities."""
    pcmc_names = {
        "Hinjewadi", "Chinchwad", "Wakad", "Pimpri", "Akurdi", "Nigdi",
        "Ravet", "Bhosari", "Moshi", "Pimple Saudagar", "Pimple Gurav",
        "Pimple Nilakh", "Sangvi", "Dapodi", "Kasarwadi",
    }

    try:
        all_gaz = load_all_gazetteers()
        combined = {}

        for city_key, localities in all_gaz.items():
            for name, meta in localities.items():
                city_name = (
                    "PCMC"
                    if city_key == "pune" and name in pcmc_names
                    else str(meta["city_key"]).capitalize()
                )

                combined[name] = {
                    "lat": float(meta["lat"]),
                    "lon": float(meta["lon"]),
                    "city": city_name,
                    "city_key": meta["city_key"],
                }

        if combined:
            return combined

    except Exception:
        pass

    return {
        lm["id"]: {
            "lat": lm["lat"],
            "lon": lm["lon"],
            "city": "Pune",
            "city_key": "pune",
        }
        for lm in LANDMARKS
    }


GAZETTEER = load_all_localities_gazetteer()

WEATHER_OPTIONS = [
    "Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "Unknown"
]
TRAFFIC_OPTIONS = ["Low", "Medium", "High", "Jam", "Unknown"]
VEHICLE_OPTIONS = ["Motorcycle", "Scooter", "Bicycle", "Electric Scooter"]
FESTIVAL_OPTIONS = ["No", "Yes", "Unknown"]
CITY_OPTIONS = ["Metropolitan", "Urban", "Semi-Urban"]
ALGORITHM_OPTIONS = ["held_karp", "hill_climbing", "exact"]

ALGORITHM_LABELS = {
    "held_karp": "Held-Karp (Exact DP)",
    "hill_climbing": "Hill Climbing (Heuristic)",
    "exact": "Exact (alias for Held-Karp)",
}

# ---------------------------------------------------------------------------
# Model Performance Analysis Helpers
# ---------------------------------------------------------------------------

PERFORMANCE_COLUMNS = [
    "Model",
    "MAE (minutes)",
    "RMSE (minutes)",
    "R2",
    "Training Time (seconds)",
    "Delay Accuracy",
    "Delay Precision",
    "Delay Recall",
    "Delay F1",
]


def load_performance_data():
    """Load model performance results generated by models/train.py."""
    if not PERFORMANCE_PATH.exists():
        return None, (
            f"Performance file not found: {PERFORMANCE_PATH}. "
            "Run `python models/train.py` from the project root first."
        )

    try:
        df = pd.read_csv(PERFORMANCE_PATH)
    except Exception as exc:
        return None, f"Could not read model performance CSV: {exc}"

    if df.empty:
        return None, "The model performance CSV contains no results."

    missing_columns = [
        column for column in PERFORMANCE_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        return None, (
            "The performance CSV is missing required columns: "
            + ", ".join(missing_columns)
            + ". Check the output format in models/train.py."
        )

    df = df.copy()

    # Convert metric columns to numeric values for tables and charts.
    numeric_columns = [
        "MAE (minutes)",
        "RMSE (minutes)",
        "R2",
        "Training Time (seconds)",
        "Delay Accuracy",
        "Delay Precision",
        "Delay Recall",
        "Delay F1",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Model"])
    df["Model"] = df["Model"].replace({"RandomForest": "Random Forest"})

    # Keep the requested model order when those models are present.
    desired_order = ["Random Forest", "XGBoost", "LightGBM"]
    df["_order"] = df["Model"].apply(
        lambda name: (
            desired_order.index(name)
            if name in desired_order
            else len(desired_order)
        )
    )
    df = df.sort_values("_order").drop(columns="_order")

    return df, None


def display_model_performance():
    """Render model comparison metrics, tables, and charts."""
    st.header("Model Performance Analysis")
    st.write(
        "Compare Random Forest, XGBoost, and LightGBM using the evaluation "
        "results generated during model training."
    )

    df, error = load_performance_data()

    if error:
        st.warning(error)
        st.info(
            "After training finishes successfully, restart or refresh the "
            "Streamlit page to view the results."
        )
        return

    st.subheader("Performance Summary")

    # Identify the best model for each metric where values are available.
    summary_specs = [
        ("MAE (minutes)", "MAE", "min"),
        ("RMSE (minutes)", "RMSE", "min"),
        ("R2", "R² Score", "max"),
        ("Training Time (seconds)", "Training Time", "min"),
        ("Delay Accuracy", "Delay Accuracy", "max"),
        ("Delay F1", "Delay F1", "max"),
    ]

    summary_items = []

    for column, label, direction in summary_specs:
        valid = df.dropna(subset=[column])

        if valid.empty:
            continue

        best_idx = (
            valid[column].idxmin()
            if direction == "min"
            else valid[column].idxmax()
        )
        best_row = valid.loc[best_idx]
        summary_items.append(
            (label, best_row["Model"], float(best_row[column]), column)
        )

    if summary_items:
        # Display six compact metric cards across two rows.
        for start in range(0, len(summary_items), 3):
            items = summary_items[start:start + 3]
            cols = st.columns(len(items))

            for col, (label, model_name, value, metric_column) in zip(cols, items):
                if metric_column in ("MAE (minutes)", "RMSE (minutes)"):
                    value_text = f"{value:.3f} min"
                elif metric_column == "Training Time (seconds)":
                    value_text = f"{value:.2f} sec"
                else:
                    value_text = f"{value:.4f}"

                with col:
                    st.metric(
                        label=f"Best {label}",
                        value=value_text,
                        help=f"Best result achieved by {model_name}.",
                    )
                    st.caption(model_name)

    st.divider()
    st.subheader("Detailed Model Comparison")

    display_df = df[PERFORMANCE_COLUMNS].copy()

    # Format the values for readability in the frontend table.
    formatters = {
        "MAE (minutes)": lambda value: (
            f"{value:.3f}" if pd.notna(value) else "N/A"
        ),
        "RMSE (minutes)": lambda value: (
            f"{value:.3f}" if pd.notna(value) else "N/A"
        ),
        "R2": lambda value: f"{value:.4f}" if pd.notna(value) else "N/A",
        "Training Time (seconds)": lambda value: (
            f"{value:.2f}" if pd.notna(value) else "N/A"
        ),
        "Delay Accuracy": lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        ),
        "Delay Precision": lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        ),
        "Delay Recall": lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        ),
        "Delay F1": lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        ),
    }

    formatted_df = display_df.copy()
    for column, formatter in formatters.items():
        formatted_df[column] = formatted_df[column].apply(formatter)

    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Lower MAE, RMSE, and training time are generally better. "
        "Higher R², delay accuracy, precision, recall, and F1 are generally better."
    )

    st.divider()
    st.subheader("Error Comparison")

    error_cols = st.columns(2)

    with error_cols[0]:
        chart_df = df.dropna(subset=["MAE (minutes)"])

        if not chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                chart_df["Model"],
                chart_df["MAE (minutes)"],
            )
            ax.set_title("Mean Absolute Error (MAE)")
            ax.set_ylabel("Error (minutes)")
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.3f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No MAE values are available.")

    with error_cols[1]:
        chart_df = df.dropna(subset=["RMSE (minutes)"])

        if not chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                chart_df["Model"],
                chart_df["RMSE (minutes)"],
            )
            ax.set_title("Root Mean Squared Error (RMSE)")
            ax.set_ylabel("Error (minutes)")
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.3f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No RMSE values are available.")

    st.divider()
    st.subheader("Model Fit and Predictive Performance")

    chart_cols = st.columns(2)

    with chart_cols[0]:
        chart_df = df.dropna(subset=["Training Time (seconds)"])

        if not chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                chart_df["Model"],
                chart_df["Training Time (seconds)"],
            )
            ax.set_title("Model Training Time")
            ax.set_ylabel("Seconds")
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.2f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No training-time values are available.")

    with chart_cols[1]:
        chart_df = df.dropna(subset=["R2"])

        if not chart_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                chart_df["Model"],
                chart_df["R2"],
            )
            ax.set_title("R² Score")
            ax.set_ylabel("R²")
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.4f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No R² values are available.")

    st.divider()
    st.subheader("Delay Classification Metrics")

    st.write(
        "These metrics evaluate delay classification using the threshold "
        "configured in the training evaluation code."
    )

    classification_columns = [
        "Delay Accuracy",
        "Delay Precision",
        "Delay Recall",
        "Delay F1",
    ]

    available_classification = [
        column
        for column in classification_columns
        if df[column].notna().any()
    ]

    if available_classification:
        plot_df = df.set_index("Model")[available_classification]

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df.plot(kind="bar", ax=ax)
        ax.set_title("Delay Classification Metrics by Model")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Metric")
        plt.xticks(rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No delay classification metrics are available.")

        # -----------------------------------------------------------------------
    # LightGBM Before vs After Optimization
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("LightGBM Before vs After Optimization")

    st.write(
        "LightGBM was selected as the best baseline model based on validation "
        "MAE and then optimized using Optuna. The comparison below shows the "
        "effect of hyperparameter optimization on the validation set."
    )

    before_after_data = {
        "Metric": [
            "MAE (minutes)",
            "RMSE (minutes)",
            "R² (%)",
            "Delay Accuracy (%)",
            "Delay F1 (%)",
        ],
        "Before Optimization": [
            3.1040,
            3.9144,
            82.60,
            93.24,
            87.12,
        ],
        "After Optimization": [
            3.0679,
            3.8612,
            83.07,
            93.27,
            87.25,
        ],
        "Improvement": [
            "1.16% ↓",
            "1.36% ↓",
            "+0.47 pp",
            "+0.03 pp",
            "+0.13 pp",
        ],
    }

    before_after_df = pd.DataFrame(before_after_data)

    st.dataframe(
        before_after_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "For MAE and RMSE, lower values are better. "
        "For R² and delay-classification metrics, higher values are better."
    )

    # -----------------------------------------------------------------------
    # Grouped Bar Chart
    # -----------------------------------------------------------------------

    st.subheader("Before vs After Optimization — Grouped Comparison")

    chart_metrics = [
        "MAE",
        "RMSE",
        "R²",
    ]

    before_values = [
        3.1040,
        3.9144,
        82.60,
    ]

    after_values = [
        3.0679,
        3.8612,
        83.07,
    ]

    x = np.arange(len(chart_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    before_bars = ax.bar(
        x - width / 2,
        before_values,
        width,
        label="Before Optimization",
    )

    after_bars = ax.bar(
        x + width / 2,
        after_values,
        width,
        label="After Optimization",
    )

    ax.set_title(
        "LightGBM Before vs After Optimization"
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.bar_label(
        before_bars,
        fmt="%.2f",
        padding=3,
    )

    ax.bar_label(
        after_bars,
        fmt="%.2f",
        padding=3,
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------------------------
    # Optuna Parameters
    # -----------------------------------------------------------------------

    with st.expander("🔧 Best LightGBM Parameters Found by Optuna"):

        optimized_parameters = {
            "n_estimators": 315,
            "learning_rate": 0.016295,
            "num_leaves": 114,
            "max_depth": 12,
            "min_child_samples": 7,
            "subsample": 0.700584,
            "colsample_bytree": 0.937420,
            "reg_alpha": 0.949480,
            "reg_lambda": 3.230862,
        }

        parameter_df = pd.DataFrame(
            list(optimized_parameters.items()),
            columns=["Parameter", "Optimized Value"],
        )

        st.dataframe(
            parameter_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Optuna evaluated 30 trials and selected the configuration "
            "with the lowest validation MAE."
        )

    st.divider()
    st.subheader("How to Interpret the Results")

    with st.expander("Metric definitions and interpretation"):
        st.markdown(
            """
            - **MAE:** Average absolute prediction error in minutes. Lower is better.
            - **RMSE:** Gives larger errors more influence than MAE. Lower is better.
            - **R²:** Indicates how much variation in delivery time the model
              explains relative to a mean-prediction baseline. Higher is generally better.
            - **Training time:** Time recorded for the model fitting step.
              It can vary with hardware and system load.
            - **Delay accuracy:** Fraction of predictions correctly classified
              by the project's delay evaluation rule.
            - **Precision:** Of predictions classified as delays, the fraction
              that are true delays.
            - **Recall:** Of actual delays, the fraction identified by the model.
            - **F1:** Harmonic mean of precision and recall.
            """
        )

    # Optional downloadable copy of the results.
    csv_bytes = df[PERFORMANCE_COLUMNS].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Performance Results (CSV)",
        data=csv_bytes,
        file_name="model_performance.csv",
        mime="text/csv",
    )

    st.caption(f"Results loaded from `{PERFORMANCE_PATH}`.")


# ---------------------------------------------------------------------------
# Time and Configuration Helpers
# ---------------------------------------------------------------------------

def get_default_ist_departure() -> tuple[date, time]:
    """Calculate default departure rounded up to the next 30 minutes."""
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


def build_default_stops_list(
    stop_ids: list[str],
    city_key: str = "bengaluru",
) -> list[dict]:
    out = []
    city_gaz = get_city_gazetteer(city_key)

    for stop_id in stop_ids:
        if stop_id in city_gaz:
            loc = city_gaz[stop_id]
            out.append({
                "id": stop_id,
                "lat": loc["lat"],
                "lon": loc["lon"],
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            })

    return out


def get_depot_coords(
    start_name: str,
    city_key: str | None = None,
) -> dict[str, float]:
    """Retrieve coordinates for a starting location from the gazetteer."""
    if city_key:
        try:
            city_gaz = get_city_gazetteer(city_key)

            if start_name in city_gaz:
                return {
                    "lat": float(city_gaz[start_name]["lat"]),
                    "lon": float(city_gaz[start_name]["lon"]),
                }
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
    departure_date, departure_time = get_default_ist_departure()
    city_meta = get_city_metadata(city_key)
    start_name = start_loc or city_meta["default_start"]
    depot_coords = get_depot_coords(start_name, city_key=city_key)

    departure_iso = (
        f"{departure_date.isoformat()}T"
        f"{departure_time.strftime('%H:%M:%S')}"
    )
    context = resolve_context(city_key, departure_date)

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
        "departure_date": departure_date,
        "departure_time": departure_time,
        "departure_iso": departure_iso,
        "rider": {
            "vehicle_type": DEFAULTS["vehicle_type"],
            "vehicle_condition": DEFAULTS["vehicle_condition"],
            "delivery_person_age": float(DEFAULTS["rider_age"]),
            "delivery_person_rating": float(DEFAULTS["rider_rating"]),
        },
        "festival": context.get("festival", DEFAULTS["festival"]),
        "city_type": context.get("city_type", DEFAULTS["city_type"]),
        "algorithm": DEFAULTS["algorithm"],
        "stops": build_default_stops_list(stop_names, city_key=city_key),
    }


def build_optimize_payload(applied_cfg: dict) -> dict:
    """Build optimize API payload using gazetteer depot coordinates."""
    city_key = applied_cfg.get("city_key", DEFAULTS["city_key"])
    start_location = applied_cfg.get(
        "start_location",
        DEFAULTS["start_location"],
    )
    coords = get_depot_coords(start_location, city_key=city_key)

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
    """Check whether code is running inside Streamlit or AppTest."""
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
    coords = {
        "depot": (depot["lat"], depot["lon"]),
        **stops_coords,
    }
    points = [coords[name] for name in route_order if name in coords]

    if center is None:
        center = (
            [
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            ]
            if points
            else [depot["lat"], depot["lon"]]
        )

    map_object = folium.Map(location=center, zoom_start=zoom_start)

    if len(points) > 1:
        folium.PolyLine(
            points,
            color=color,
            weight=4,
            opacity=0.85,
        ).add_to(map_object)

    depot_html = (
        '<div style="font-size:10px;font-weight:bold;color:white;'
        'background:#C62828;border-radius:4px;padding:2px 8px;'
        'border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,.5);">'
        'DEPOT</div>'
    )

    folium.Marker(
        [depot["lat"], depot["lon"]],
        tooltip="Depot (Start)",
        icon=folium.DivIcon(html=depot_html),
    ).add_to(map_object)

    for index, node in enumerate(route_order):
        if node == "depot" or node not in coords:
            continue

        badge = (
            f'<div style="font-size:11px;font-weight:bold;color:white;'
            f'background:{color};border-radius:50%;width:24px;height:24px;'
            'display:flex;align-items:center;justify-content:center;'
            'border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,.5);">'
            f'{index}</div>'
        )

        folium.Marker(
            coords[node],
            tooltip=f"{index}. {node}",
            icon=folium.DivIcon(html=badge),
        ).add_to(map_object)

    return map_object


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def run_app():
    # State initialization
    if "applied_config" not in st.session_state:
        st.session_state["applied_config"] = build_default_applied_config()

    if "stop_configs" not in st.session_state:
        st.session_state["stop_configs"] = {}

    cities_data = load_cities_data()
    city_options = ["bengaluru", "pune", "mumbai", "delhi"]
    city_labels = {
        key: cities_data[key]["display_name"]
        for key in city_options
    }

    # City change callback
    def on_city_change():
        chosen_city = st.session_state.get(
            "city_select_key",
            "bengaluru",
        )
        meta = get_city_metadata(chosen_city)
        default_start = meta["default_start"]

        st.session_state["start_loc_key"] = default_start
        st.session_state["stops_multiselect_key"] = []
        st.session_state["stop_configs"] = {}

        st.session_state.pop("optimize_result", None)
        st.session_state.pop("last_optimized_payload", None)
        st.session_state.pop("last_optimized_config", None)

        st.session_state["applied_config"] = build_default_applied_config(
            city_key=chosen_city,
            start_loc=default_start,
            stops=[],
        )

    # Reset callback
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

    # Page setup
    st.set_page_config(
        page_title="Delivery ETA & Route Optimizer",
        page_icon="🚴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚴 Delivery ETA Predictor & Route Optimizer")
    st.caption(
        "Multi-City Route Optimization · ETA Prediction · Held-Karp Exact Solver"
    )

    # City selection
    st.sidebar.subheader("🏙️ City Selection")

    current_city_key = st.session_state.get(
        "city_select_key",
        DEFAULTS["city_key"],
    )
    city_index = (
        city_options.index(current_city_key)
        if current_city_key in city_options
        else 0
    )

    selected_city = st.sidebar.selectbox(
        "City",
        options=city_options,
        format_func=lambda key: city_labels[key],
        index=city_index,
        key="city_select_key",
        on_change=on_city_change,
        help="Select the operational city.",
    )

    current_city_meta = get_city_metadata(selected_city)
    current_city_gaz = get_city_gazetteer(selected_city)
    city_locality_names = list(current_city_gaz.keys())

    # Starting location
    st.sidebar.subheader("📍 Starting Location (Depot)")

    default_start_value = st.session_state.get(
        "start_loc_key",
        current_city_meta["default_start"],
    )

    if default_start_value not in city_locality_names:
        default_start_value = current_city_meta["default_start"]
        st.session_state["start_loc_key"] = default_start_value

    start_index = (
        city_locality_names.index(default_start_value)
        if default_start_value in city_locality_names
        else 0
    )

    start_location = st.sidebar.selectbox(
        "Starting location",
        options=city_locality_names,
        index=start_index,
        key="start_loc_key",
        help="Select the depot or origin location.",
    )

    depot_info = current_city_gaz[start_location]

    st.sidebar.caption(
        f"Coordinates: `{depot_info['lat']:.4f}° N, "
        f"{depot_info['lon']:.4f}° E` "
        f"({current_city_meta['display_name']})"
    )

    current_applied = st.session_state.get("applied_config")

    if current_applied and (
        current_applied.get("start_location") != start_location
        or current_applied.get("city_key") != selected_city
    ):
        current_applied["city_key"] = selected_city
        current_applied["city_display"] = current_city_meta["display_name"]
        current_applied["start_location"] = start_location
        current_applied["depot_lat"] = float(depot_info["lat"])
        current_applied["depot_lon"] = float(depot_info["lon"])

    # Stop selection
    st.sidebar.subheader("🗺️ Stops Configuration")

    available_stops = [
        name for name in city_locality_names
        if name != start_location
    ]

    current_stops = st.session_state.get(
        "stops_multiselect_key",
        list(DEFAULTS["stops"]) if selected_city == "bengaluru" else [],
    )

    current_stops = [
        stop for stop in current_stops
        if stop in available_stops
    ]
    st.session_state["stops_multiselect_key"] = current_stops

    selected_stops = st.sidebar.multiselect(
        "Select stops",
        options=available_stops,
        default=current_stops,
        key="stops_multiselect_key",
        max_selections=12,
        help="Select between 2 and 12 delivery stops.",
    )

    st.sidebar.caption(
        "Select 2 to 12 stops. Held-Karp exact solver runs in "
        "O(n²·2ⁿ). At least 2 stops are required."
    )

    # Random stop selection
    def sample_random_stops_callback():
        available = [
            name for name in city_locality_names
            if name != start_location
        ]
        st.session_state["stops_multiselect_key"] = random.sample(
            available,
            min(4, len(available)),
        )

    st.sidebar.button(
        "🔀 Sample Random Stops",
        on_click=sample_random_stops_callback,
        help="Populate the route with random stops.",
    )

    st.sidebar.divider()

    # Configuration form
    with st.sidebar.form("config_form"):
        st.subheader("⏰ Departure Time (IST)")

        now_ist = datetime.now(IST_ZONE)
        default_date, default_time = get_default_ist_departure()

        dep_date = st.date_input(
            "Departure Date",
            value=st.session_state.get("dep_date_key", default_date),
            min_value=now_ist.date(),
            key="dep_date_key",
            help="Must be today or a future date in IST.",
        )

        dep_time = st.time_input(
            "Departure Time",
            value=st.session_state.get("dep_time_key", default_time),
            key="dep_time_key",
            help="Departure time in IST.",
        )

        is_past_time = (
            dep_date == now_ist.date()
            and dep_time < now_ist.time().replace(tzinfo=None)
        )

        if is_past_time:
            st.error(
                f"Selected time ({dep_time.strftime('%H:%M')}) is earlier "
                f"than current IST time ({now_ist.strftime('%H:%M')})."
            )

        st.divider()
        st.subheader("🧑 Rider Details")

        vehicle_type = st.selectbox(
            "Vehicle Type",
            VEHICLE_OPTIONS,
            index=VEHICLE_OPTIONS.index(
                st.session_state.get(
                    "vehicle_type_key",
                    DEFAULTS["vehicle_type"],
                )
            ),
            key="vehicle_type_key",
        )

        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            [0, 1, 2, 3],
            index=st.session_state.get(
                "vehicle_condition_key",
                DEFAULTS["vehicle_condition"],
            ),
            key="vehicle_condition_key",
            help="0 = poor, 3 = excellent.",
        )

        rider_age = st.slider(
            "Rider Age",
            min_value=20,
            max_value=39,
            value=int(
                st.session_state.get("rider_age_key", DEFAULTS["rider_age"])
            ),
            key="rider_age_key",
        )

        rider_rating = st.slider(
            "Rider Rating",
            min_value=2.5,
            max_value=5.0,
            value=float(
                st.session_state.get(
                    "rider_rating_key",
                    DEFAULTS["rider_rating"],
                )
            ),
            step=0.1,
            key="rider_rating_key",
        )

        st.divider()
        st.subheader("⚙️ Route Settings")

        resolved_context = resolve_context(selected_city, dep_date)

        festival_default = st.session_state.get(
            "festival_key",
            resolved_context["festival"],
        )
        if festival_default not in FESTIVAL_OPTIONS:
            festival_default = resolved_context["festival"]

        city_type_default = st.session_state.get(
            "city_type_key",
            resolved_context["city_type"],
        )
        if city_type_default not in CITY_OPTIONS:
            city_type_default = resolved_context["city_type"]

        festival = st.selectbox(
            "Festival Day",
            FESTIVAL_OPTIONS,
            index=FESTIVAL_OPTIONS.index(festival_default),
            key="festival_key",
        )

        city_type = st.selectbox(
            "City Type",
            CITY_OPTIONS,
            index=CITY_OPTIONS.index(city_type_default),
            key="city_type_key",
        )

        algorithm_default = st.session_state.get(
            "algorithm_key",
            DEFAULTS["algorithm"],
        )
        if algorithm_default not in ALGORITHM_OPTIONS:
            algorithm_default = DEFAULTS["algorithm"]

        algorithm = st.selectbox(
            "Optimizer Algorithm",
            ALGORITHM_OPTIONS,
            format_func=lambda key: ALGORITHM_LABELS[key],
            index=ALGORITHM_OPTIONS.index(algorithm_default),
            key="algorithm_key",
        )

        st.divider()
        st.subheader("📋 Stop Conditions Editor")

        editor_rows = []

        for stop_id in selected_stops:
            if stop_id in current_city_gaz:
                loc = current_city_gaz[stop_id]
                config = st.session_state["stop_configs"].get(stop_id, {})

                editor_rows.append({
                    "id": stop_id,
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "weather": config.get("weather", "Sunny"),
                    "traffic": config.get("traffic", "Medium"),
                    "multiple_deliveries": int(
                        config.get("multiple_deliveries", 0)
                    ),
                })

        editor_df = pd.DataFrame(editor_rows)

        if not editor_df.empty:
            edited_table = st.data_editor(
                editor_df,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn(
                        "Stop ID",
                        disabled=True,
                    ),
                    "lat": st.column_config.NumberColumn(
                        "Lat",
                        format="%.4f",
                        disabled=True,
                    ),
                    "lon": st.column_config.NumberColumn(
                        "Lon",
                        format="%.4f",
                        disabled=True,
                    ),
                    "weather": st.column_config.SelectboxColumn(
                        "Weather",
                        options=WEATHER_OPTIONS,
                    ),
                    "traffic": st.column_config.SelectboxColumn(
                        "Traffic",
                        options=TRAFFIC_OPTIONS,
                    ),
                    "multiple_deliveries": st.column_config.NumberColumn(
                        "Multi-Del",
                        min_value=0,
                        max_value=3,
                        step=1,
                    ),
                },
                key="stops_table_editor",
            )
        else:
            edited_table = editor_df
            st.caption("No stops currently selected.")

        st.divider()

        apply_clicked = st.form_submit_button(
            "✅ Apply Filters",
            type="primary",
            use_container_width=True,
        )

        st.form_submit_button(
            "🔄 Reset to Defaults",
            use_container_width=True,
            on_click=reset_to_defaults,
        )

    # Handle form submission
    if apply_clicked:
        if isinstance(edited_table, pd.DataFrame) and not edited_table.empty:
            for _, row in edited_table.iterrows():
                st.session_state["stop_configs"][row["id"]] = {
                    "weather": str(row["weather"]),
                    "traffic": str(row["traffic"]),
                    "multiple_deliveries": int(row["multiple_deliveries"]),
                }

        now_check = datetime.now(IST_ZONE)
        current_ist_time = now_check.time().replace(tzinfo=None)

        if (
            dep_date == now_check.date()
            and dep_time < current_ist_time
        ):
            st.sidebar.error(
                "Cannot apply: departure time cannot be earlier than current IST time."
            )
        elif len(selected_stops) < 2:
            st.sidebar.error(
                "Cannot apply: at least 2 delivery stops are required."
            )
        elif len(selected_stops) > 12:
            st.sidebar.error(
                f"Cannot apply: at most 12 stops allowed; "
                f"current count is {len(selected_stops)}."
            )
        else:
            stops_payload = []

            for stop_id in selected_stops:
                if stop_id in current_city_gaz:
                    loc = current_city_gaz[stop_id]
                    config = st.session_state["stop_configs"].get(
                        stop_id,
                        {},
                    )

                    stops_payload.append({
                        "id": stop_id,
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "weather": config.get("weather", "Sunny"),
                        "traffic": config.get("traffic", "Medium"),
                        "multiple_deliveries": int(
                            config.get("multiple_deliveries", 0)
                        ),
                    })

            departure_iso = (
                f"{dep_date.isoformat()}T{dep_time.strftime('%H:%M:%S')}"
            )
            depot_coords = get_depot_coords(
                start_location,
                city_key=selected_city,
            )

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
                "stops": stops_payload,
            }

            st.sidebar.success(
                f"Filters applied for {current_city_meta['display_name']}!"
            )

    # Tabs
    tab_route, tab_demo, tab_model = st.tabs([
        "Route Optimizer",
        "Local Optima Demo",
        "Model Comparison",
    ])

    # =======================================================================
    # Tab 1 — Route Optimizer
    # =======================================================================

    with tab_route:
        applied_cfg = st.session_state.get("applied_config", {})

        if applied_cfg:
            city_display = applied_cfg.get(
                "city_display",
                current_city_meta["display_name"],
            )
            city_type_value = applied_cfg.get("city_type", "Metropolitan")
            festival_value = applied_cfg.get("festival", "No")

            st.info(
                f"**Detected context:** {city_display} · "
                f"{city_type_value} · Festival: {festival_value}"
            )

            with st.expander("🔍 Detailed Applied Parameters"):
                col1, col2, col3 = st.columns(3)

                col1.markdown(
                    f"**Start (Depot):** "
                    f"{applied_cfg.get('start_location')}  \n"
                    f"Coordinates: {applied_cfg.get('depot_lat', 0.0):.4f}, "
                    f"{applied_cfg.get('depot_lon', 0.0):.4f}"
                )

                col2.markdown(
                    f"**Departure (IST):** "
                    f"{applied_cfg.get('departure_iso', '').replace('T', ' ')}  \n"
                    f"City Type: {city_type_value} · Festival: {festival_value}"
                )

                rider = applied_cfg.get("rider", {})

                col3.markdown(
                    f"**Rider & Solver:** {rider.get('vehicle_type')} · "
                    f"{ALGORITHM_LABELS.get(applied_cfg.get('algorithm', ''), applied_cfg.get('algorithm', ''))}  \n"
                    f"Age: {rider.get('delivery_person_age', 0):.0f} · "
                    f"Rating: {rider.get('delivery_person_rating', 0):.1f}★"
                )

        last_cfg = st.session_state.get("last_optimized_config")
        result = st.session_state.get("optimize_result")

        if result and last_cfg and last_cfg != applied_cfg:
            st.warning(
                "Settings have changed since the last run. "
                "The displayed results belong to the previous configuration. "
                "Click Optimize Route to recalculate."
            )

        stops_list = applied_cfg.get("stops", [])
        now_ist = datetime.now(IST_ZONE)

        is_time_past = False
        if (
            "departure_date" in applied_cfg
            and "departure_time" in applied_cfg
        ):
            is_time_past = (
                applied_cfg["departure_date"] == now_ist.date()
                and applied_cfg["departure_time"]
                < now_ist.time().replace(tzinfo=None)
            )

        can_optimize = 2 <= len(stops_list) <= 12 and not is_time_past

        button_col, info_col = st.columns([1, 3])

        with button_col:
            clicked = st.button(
                "🚀 Optimize Route",
                type="primary",
                disabled=not can_optimize,
                use_container_width=True,
            )

        with info_col:
            if is_time_past:
                st.error(
                    "Departure time is earlier than the current IST time."
                )
            elif len(stops_list) < 2:
                st.info(
                    "Select and apply at least 2 delivery stops before optimizing."
                )
            elif not can_optimize:
                st.info("Fix the validation issues before optimizing.")
            else:
                st.success(
                    f"Ready to optimize {len(stops_list)} stop(s) in "
                    f"{applied_cfg.get('city_display', selected_city.title())} · "
                    f"{ALGORITHM_LABELS.get(applied_cfg.get('algorithm', 'held_karp'))}"
                )

        if clicked:
            current_payload = build_optimize_payload(applied_cfg)

            with st.spinner("Running ETA prediction and route optimization..."):
                try:
                    response = requests.post(
                        f"{API_URL}/optimize",
                        json=current_payload,
                        timeout=60,
                    )

                    if response.status_code == 200:
                        st.session_state["optimize_result"] = response.json()
                        st.session_state["last_optimized_payload"] = current_payload
                        st.session_state["last_optimized_config"] = copy.deepcopy(
                            applied_cfg
                        )

                        result = st.session_state["optimize_result"]

                    elif response.status_code == 422:
                        detail = response.json().get("detail", response.text)

                        errors = (
                            [
                                f"**{' → '.join(str(x) for x in item.get('loc', []) if x != 'body')}**: "
                                f"{item.get('msg', 'Invalid value')}"
                                for item in detail
                            ]
                            if isinstance(detail, list)
                            else [str(detail)]
                        )

                        st.error(
                            "Validation Error (422):\n\n"
                            + "\n".join(errors)
                        )

                    else:
                        st.error(
                            f"API Error ({response.status_code}): {response.text}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Cannot reach the API at {API_URL}.\n\n"
                        "Start the backend using:\n"
                        "`python -m uvicorn api.main:app --reload`"
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "Request timed out after 60 seconds. "
                        "Try fewer stops or a simpler algorithm."
                    )

                except requests.exceptions.RequestException as exc:
                    st.error(f"Request failed: {exc}")

        opt_cfg = st.session_state.get("last_optimized_config", applied_cfg)
        opt_city_key = opt_cfg.get("city_key", selected_city)
        opt_city_meta = get_city_metadata(opt_city_key)

        map_center = [
            opt_city_meta["center_lat"],
            opt_city_meta["center_lon"],
        ]
        map_zoom = opt_city_meta.get("map_zoom", 12)

        map_depot_coords = get_depot_coords(
            opt_cfg.get("start_location", opt_city_meta["default_start"]),
            city_key=opt_city_key,
        )

        result = st.session_state.get("optimize_result")

        if result:
            st.divider()

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric(
                "Algorithm",
                ALGORITHM_LABELS.get(
                    result.get("algorithm", ""),
                    result.get("algorithm", ""),
                ),
            )
            metric2.metric(
                "Baseline Duration",
                f"{result['baseline_total_minutes']:.1f} min",
            )
            metric3.metric(
                "Optimized Duration",
                f"{result['total_minutes']:.1f} min",
            )

            improvement = result.get("improvement_pct", 0)
            metric4.metric(
                "Improvement",
                f"{improvement:.2f}%",
                delta=(
                    f"{improvement:+.2f}%"
                    if improvement != 0
                    else None
                ),
                delta_color="normal" if improvement >= 0 else "inverse",
            )

            distance_warnings = result.get("distance_warnings", {})

            if distance_warnings.get("count", 0) > 0:
                training_range = distance_warnings.get("training_range_km", [])
                if len(training_range) >= 2:
                    st.warning(
                        f"{distance_warnings['count']} leg(s) had distances "
                        f"outside the training range "
                        f"({training_range[0]}–{training_range[1]} km) "
                        "and were clipped. Predictions for these legs "
                        "may be less reliable."
                    )

            route_col1, route_col2 = st.columns(2)

            with route_col1:
                st.markdown(
                    f"**Baseline Route ({opt_city_meta['display_name']})**"
                )
                st.write(" → ".join(result.get("baseline_route", [])))

            with route_col2:
                st.markdown(
                    f"**Optimized Route ({opt_city_meta['display_name']})**"
                )
                st.write(" → ".join(result.get("route", [])))

            stop_coords = {
                stop["id"]: (stop["lat"], stop["lon"])
                for stop in result.get("stops", [])
            }

            before_col, after_col = st.columns(2)

            with before_col:
                st.subheader(
                    f"Before: Baseline Route ({opt_city_meta['display_name']})"
                )

                map_before = build_map(
                    map_depot_coords,
                    stop_coords,
                    result["baseline_route"],
                    color="#757575",
                    center=map_center,
                    zoom_start=map_zoom,
                )

                st_folium(
                    map_before,
                    width=None,
                    height=420,
                    returned_objects=[],
                    use_container_width=True,
                    key="map_before",
                )

            with after_col:
                st.subheader(
                    f"After: Optimized Route ({opt_city_meta['display_name']})"
                )

                map_after = build_map(
                    map_depot_coords,
                    stop_coords,
                    result["route"],
                    color="#2E7D32",
                    center=map_center,
                    zoom_start=map_zoom,
                )

                st_folium(
                    map_after,
                    width=None,
                    height=420,
                    returned_objects=[],
                    use_container_width=True,
                    key="map_after",
                )

            st.subheader(
                f"📋 Per-Stop Delivery Schedule ({opt_city_meta['display_name']})"
            )

            if result.get("stops"):
                schedule_df = pd.DataFrame(result["stops"])

                rename_columns = {
                    "id": "Stop ID",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "leg_minutes": "Leg (min)",
                    "eta_minutes": "ETA (min)",
                    "arrival_time": "Arrival Time",
                }

                display_columns = [
                    column
                    for column in rename_columns
                    if column in schedule_df.columns
                ]

                schedule_df = schedule_df[display_columns].rename(
                    columns=rename_columns
                )

                st.dataframe(
                    schedule_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("🔍 Full API Response (JSON)"):
                st.json(result)

    # =======================================================================
    # Tab 2 — Local Optima Demo
    # =======================================================================

    with tab_demo:
        st.caption(
            "Hill climbing from random starting routes. Costs come from a "
            "synthetic asymmetric time matrix with a fixed seed."
        )

        starts = st.slider(
            "Number of random starts",
            10,
            100,
            50,
            key="demo_starts",
        )

        demo_seed = st.number_input(
            "Random Seed",
            value=42,
            step=1,
            key="demo_seed_val",
        )

        demo_n = max(len(applied_cfg.get("stops", [])) + 1, 4)

        demo = run_local_optima_demo(
            num_starts=starts,
            n_stops=demo_n,
            seed=int(demo_seed),
        )

        costs = [item["final_cost"] for item in demo]

        if costs:
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                range(1, len(costs) + 1),
                costs,
                marker="o",
                linewidth=1,
                markersize=4,
                label="Final hill-climbing cost",
            )

            ax.axhline(
                min(costs),
                linestyle="--",
                color="green",
                label=f"Best = {min(costs):.2f} min",
            )

            ax.axhline(
                max(costs),
                linestyle="--",
                color="red",
                label=f"Worst = {max(costs):.2f} min",
            )

            ax.set_xlabel("Random starting route")
            ax.set_ylabel("Final route cost (minutes)")
            ax.set_title(
                "Hill Climbing — Effect of Starting Route on Final Cost"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            unique_costs = len({
                round(cost, 6) for cost in costs
            })

            st.write(
                f"**{unique_costs}** unique final costs out of {len(costs)} "
                f"runs. Duplicate final costs: "
                f"**{len(costs) - unique_costs}**."
            )

    # =======================================================================
    # Tab 3 — Model Comparison and Performance Analysis
    # =======================================================================

    with tab_model:
        display_model_performance()

        st.divider()
        st.subheader("Model Comparison Report")

        if COMPARISON_PATH.exists():
            try:
                content = COMPARISON_PATH.read_text(encoding="utf-8")

                # Remove the specified section and everything after it,
                # if that heading is present in the generated report.
                report_marker = "## Key Findings & Viva Talking Points"

                if report_marker in content:
                    content = content.split(report_marker)[0].strip()

                st.markdown(content)
            except OSError as exc:
                st.warning(f"Could not read the model comparison report: {exc}")
        else:
            st.info(
                "`docs/model_comparison.md` has not been generated yet. "
                "Run `python models/train.py` to create it."
            )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if is_running_in_streamlit():
    run_app()


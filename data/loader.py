"""Loader module for cities, gazetteers, and context resolution."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
CITIES_FILE = DATA_DIR / "cities.csv"
GAZETTEER_DIR = DATA_DIR / "gazetteer"
CITY_TYPES_FILE = DATA_DIR / "city_types.csv"
FESTIVALS_FILE = DATA_DIR / "festivals_india.csv"

# Valid city types accepted by the 27-feature model feature builder
VALID_CITY_TYPES = {"Metropolitan", "Urban", "Semi-Urban"}

# Bounding box for India
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 98.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


@lru_cache(maxsize=1)
def load_cities_data() -> dict[str, dict[str, Any]]:
    """Load and validate data/cities.csv."""
    if not CITIES_FILE.exists():
        raise FileNotFoundError(f"Cities metadata file not found at {CITIES_FILE}")

    df = pd.read_csv(CITIES_FILE)
    cities: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        key = str(row["city_key"]).strip().lower()
        city_type = str(row["city_type"]).strip()

        if city_type not in VALID_CITY_TYPES:
            raise ValueError(
                f"Invalid city_type '{city_type}' for city '{key}'. "
                f"Must be one of {sorted(VALID_CITY_TYPES)}"
            )

        center_lat = float(row["center_lat"])
        center_lon = float(row["center_lon"])

        if not (INDIA_LAT_MIN <= center_lat <= INDIA_LAT_MAX and INDIA_LON_MIN <= center_lon <= INDIA_LON_MAX):
            raise ValueError(f"City center for '{key}' is outside India bounding box: ({center_lat}, {center_lon})")

        cities[key] = {
            "city_key": key,
            "display_name": str(row["display_name"]).strip(),
            "state": str(row["state"]).strip(),
            "city_type": city_type,
            "default_start": str(row["default_start"]).strip(),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "map_zoom": int(row.get("map_zoom", 12)),
            "service_radius_km": float(row.get("service_radius_km", 25.0)),
        }

    return cities


@lru_cache(maxsize=1)
def load_all_gazetteers() -> dict[str, dict[str, dict[str, Any]]]:
    """Load, validate, and cache all gazetteer CSV files per city.

    Returns:
        dict mapping city_key -> dict of locality_name -> {lat, lon, city_key}
    """
    cities = load_cities_data()
    gazetteers: dict[str, dict[str, dict[str, Any]]] = {}

    for city_key, city_meta in cities.items():
        csv_path = GAZETTEER_DIR / f"{city_key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Gazetteer file missing for city '{city_key}' at {csv_path}")

        df = pd.read_csv(csv_path)
        city_localities: dict[str, dict[str, Any]] = {}
        seen_names: set[str] = set()
        outliers: list[str] = []

        c_lat = city_meta["center_lat"]
        c_lon = city_meta["center_lon"]

        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            if not name:
                continue

            # Check 1: No duplicate names within a city
            if name in seen_names:
                raise ValueError(f"Duplicate locality name '{name}' detected in city '{city_key}' gazetteer.")
            seen_names.add(name)

            lat = float(row["lat"])
            lon = float(row["lon"])

            # Check 2: Lat/lon inside India bounding box
            if not (INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX and INDIA_LON_MIN <= lon <= INDIA_LON_MAX):
                raise ValueError(
                    f"Locality '{name}' in city '{city_key}' has coordinates ({lat}, {lon}) "
                    f"outside India bounding box."
                )

            # Check 3: Check distance to center within ~40 km
            dist_km = haversine_km(c_lat, c_lon, lat, lon)
            if dist_km > 40.0:
                outliers.append(f"{name} ({dist_km:.1f} km)")

            city_localities[name] = {
                "name": name,
                "lat": lat,
                "lon": lon,
                "city_key": city_key,
                "dist_to_center_km": dist_km,
            }

        if outliers:
            logger.warning(
                f"City '{city_key}' has {len(outliers)} localities farther than 40 km from center: "
                f"{', '.join(outliers)}"
            )

        # Check 4: default_start exists in that city's gazetteer
        default_start = city_meta["default_start"]
        if default_start not in city_localities:
            raise ValueError(
                f"Default start '{default_start}' for city '{city_key}' does not exist in its gazetteer!"
            )

        gazetteers[city_key] = city_localities

    return gazetteers


def get_city_metadata(city_key: str = "bengaluru") -> dict[str, Any]:
    """Get metadata for a specific city."""
    key = str(city_key).strip().lower()
    cities = load_cities_data()
    if key not in cities:
        key = "bengaluru"
    return cities[key]


def get_city_gazetteer(city_key: str = "bengaluru") -> dict[str, dict[str, Any]]:
    """Get localities for a specific city."""
    key = str(city_key).strip().lower()
    all_gaz = load_all_gazetteers()
    if key not in all_gaz:
        key = "bengaluru"
    return all_gaz[key]


@lru_cache(maxsize=1)
def load_festivals_data() -> pd.DataFrame:
    """Load festivals data."""
    if FESTIVALS_FILE.exists():
        df = pd.read_csv(FESTIVALS_FILE)
        df["date_str"] = df["date"].astype(str).str.strip()
        df["region"] = df["region"].astype(str).str.strip()
        return df
    return pd.DataFrame(columns=["date", "festival_name", "region", "verified", "date_str"])


def resolve_context(city_key: str = "bengaluru", departure_time: datetime | date | str | None = None) -> dict[str, str]:
    """Resolve city_type and festival status for a city and departure time.

    The festival lookup for a route uses the city's state plus national rows.
    """
    city_meta = get_city_metadata(city_key)
    city_type = city_meta["city_type"]
    state = city_meta["state"]

    target_date_str = None
    if isinstance(departure_time, datetime):
        target_date_str = departure_time.date().isoformat()
    elif isinstance(departure_time, date):
        target_date_str = departure_time.isoformat()
    elif isinstance(departure_time, str):
        target_date_str = departure_time.split("T")[0].split(" ")[0].strip()

    is_festival = "No"
    festival_name = None

    if target_date_str:
        festivals_df = load_festivals_data()
        if not festivals_df.empty:
            matches = festivals_df[
                (festivals_df["date_str"] == target_date_str)
                & (festivals_df["region"].str.lower().isin(["national", state.lower()]))
            ]
            if not matches.empty:
                is_festival = "Yes"
                festival_name = str(matches.iloc[0]["festival_name"])

    return {
        "city_type": city_type,
        "festival": is_festival,
        "festival_name": festival_name or "",
    }


def validate_coordinates_for_city(
    city_key: str,
    depot: tuple[float, float],
    stops: list[tuple[str, float, float]],
) -> list[str]:
    """Validate that depot and stops are within service_radius_km of the city center.

    Returns a list of error strings for any violations.
    """
    city_meta = get_city_metadata(city_key)
    c_lat = city_meta["center_lat"]
    c_lon = city_meta["center_lon"]
    radius = city_meta["service_radius_km"]
    city_name = city_meta["display_name"]

    violations: list[str] = []

    # Check depot
    depot_dist = haversine_km(c_lat, c_lon, depot[0], depot[1])
    if depot_dist > radius:
        violations.append(
            f"Depot location ({depot[0]:.4f}, {depot[1]:.4f}) is {depot_dist:.1f} km from {city_name} center, "
            f"exceeding the service radius of {radius:.1f} km."
        )

    # Check stops
    for stop_id, s_lat, s_lon in stops:
        s_dist = haversine_km(c_lat, c_lon, s_lat, s_lon)
        if s_dist > radius:
            violations.append(
                f"Stop '{stop_id}' ({s_lat:.4f}, {s_lon:.4f}) is {s_dist:.1f} km from {city_name} center, "
                f"exceeding the service radius of {radius:.1f} km."
            )

    return violations

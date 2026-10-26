"""Tests for frontend helpers and invariants."""

from frontend.app import GAZETTEER, LANDMARKS, build_map

# Pune / Indian city bounding box (matches the actual landmarks)
PUNE_LAT_MIN = 18.40
PUNE_LAT_MAX = 18.68
PUNE_LON_MIN = 73.68
PUNE_LON_MAX = 73.99


def test_landmarks_count_and_uniqueness():
    assert len(LANDMARKS) == 15
    ids = [lm["id"] for lm in LANDMARKS]
    assert len(ids) == len(set(ids)), "Landmark IDs must be unique"


def test_landmarks_within_pune_box():
    for lm in LANDMARKS:
        assert PUNE_LAT_MIN <= lm["lat"] <= PUNE_LAT_MAX, f"{lm['id']} lat out of bounds"
        assert PUNE_LON_MIN <= lm["lon"] <= PUNE_LON_MAX, f"{lm['id']} lon out of bounds"


def test_gazetteer_loaded_and_valid():
    assert len(GAZETTEER) >= 40, f"Expected at least 40 localities, got {len(GAZETTEER)}"
    required = [
        "Kothrud", "Katraj", "Hinjewadi", "Hadapsar", "Shivajinagar",
        "Pune Station", "Chinchwad", "Baner", "Wakad", "Viman Nagar", "Kharadi"
    ]
    for req in required:
        assert req in GAZETTEER, f"Missing required locality: {req}"
        loc = GAZETTEER[req]
        assert PUNE_LAT_MIN <= loc["lat"] <= PUNE_LAT_MAX, f"{req} lat out of bounds"
        assert PUNE_LON_MIN <= loc["lon"] <= PUNE_LON_MAX, f"{req} lon out of bounds"


def test_central_pune_default_coordinates():
    assert "Shivajinagar" in GAZETTEER
    loc = GAZETTEER["Shivajinagar"]
    assert abs(loc["lat"] - 18.5204) < 1e-4
    assert abs(loc["lon"] - 73.8567) < 1e-4


def test_build_map_creates_folium_map():
    depot = {"lat": 18.5204, "lon": 73.8567}
    stops_coords = {
        "s1": (18.5362, 73.8938),
        "s2": (18.5308, 73.8474),
    }
    route = ["depot", "s1", "s2"]
    m = build_map(depot, stops_coords, route)
    assert m is not None
    html = m.get_root().render()
    assert "DEPOT" in html
    assert "s1" in html
    assert "s2" in html


def test_app_test_initial_load():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run()
    assert not at.exception
    assert at.title[0].value == "\U0001f6b4 Delivery ETA Predictor & Route Optimizer"
    assert [t.label for t in at.tabs] == [
        "Route Optimizer",
        "Local Optima Demo",
        "Model Comparison",
    ]


def test_depot_payload_strictly_follows_selected_start():
    """Regression test: for each city, selecting 3 different starts produces an optimize

    payload whose depot lat/lon equals that gazetteer row exactly.
    """
    from frontend.app import build_default_applied_config, build_optimize_payload, get_depot_coords

    city_samples = {
        "Pune": ["Shivajinagar", "Kothrud", "Aundh"],
        "PCMC": ["Hinjewadi", "Chinchwad", "Wakad"],
    }

    base_config = build_default_applied_config()

    for city, starts in city_samples.items():
        payloads = []
        for start in starts:
            assert start in GAZETTEER, f"{start} missing from GAZETTEER"
            assert GAZETTEER[start]["city"] == city, f"{start} expected to be in {city}"

            cfg = dict(base_config)
            cfg["start_location"] = start

            # Direct helper verification
            coords = get_depot_coords(start)
            assert coords["lat"] == GAZETTEER[start]["lat"]
            assert coords["lon"] == GAZETTEER[start]["lon"]

            # Optimize payload builder verification
            payload = build_optimize_payload(cfg)
            assert payload["depot_lat"] == GAZETTEER[start]["lat"]
            assert payload["depot_lon"] == GAZETTEER[start]["lon"]
            payloads.append(payload)

        # Assert payload depot coordinates change when start changes
        assert (payloads[0]["depot_lat"], payloads[0]["depot_lon"]) != (payloads[1]["depot_lat"], payloads[1]["depot_lon"])
        assert (payloads[1]["depot_lat"], payloads[1]["depot_lon"]) != (payloads[2]["depot_lat"], payloads[2]["depot_lon"])
        assert (payloads[0]["depot_lat"], payloads[0]["depot_lon"]) != (payloads[2]["depot_lat"], payloads[2]["depot_lon"])


def test_app_test_select_starting_location():
    """Test that changing start location in AppTest updates selectbox without exceptions."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run()
    assert not at.exception
    assert at.selectbox(key="city_select_key").value == "bengaluru"

    # Select a different starting location in Bengaluru
    at.selectbox(key="start_loc_key").select("Koramangala").run()
    assert not at.exception
    assert at.selectbox(key="start_loc_key").value == "Koramangala"


def test_city_selector_switches_localities():
    """Test that switching city in AppTest updates available start and stop localities."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run()
    assert not at.exception

    # Switch city to Mumbai
    at.selectbox(key="city_select_key").select("mumbai").run()
    assert not at.exception
    assert at.selectbox(key="city_select_key").value == "mumbai"
    assert at.selectbox(key="start_loc_key").value == "Dadar"

    # Verify options only contain Mumbai localities
    start_options = at.selectbox(key="start_loc_key").options
    assert "Dadar" in start_options
    assert "Bandra West" in start_options
    assert "Indiranagar" not in start_options
    assert "Shivajinagar" not in start_options


def test_gazetteer_loader_cities():
    """Test data.loader city metadata and gazetteers."""
    from data.loader import load_cities_data, load_all_gazetteers

    cities = load_cities_data()
    assert set(cities.keys()) == {"pune", "mumbai", "bengaluru", "delhi"}

    gaz = load_all_gazetteers()
    for ckey in ["pune", "mumbai", "bengaluru", "delhi"]:
        assert len(gaz[ckey]) == 25, f"Expected 25 localities for {ckey}, got {len(gaz[ckey])}"
        assert cities[ckey]["default_start"] in gaz[ckey]
        assert cities[ckey]["city_type"] == "Metropolitan"





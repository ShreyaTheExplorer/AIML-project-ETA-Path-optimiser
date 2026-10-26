from datetime import datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


BASE_RIDER = {
    "vehicle_type": "Motorcycle",
    "vehicle_condition": 2,
    "delivery_person_age": 25,
    "delivery_person_rating": 4.5,
}

BASE_REQUEST = {
    "depot_lat": 18.5204,
    "depot_lon": 73.8567,
    "rider": BASE_RIDER,
    "festival": "No",
    "city_type": "Metropolitan",
    "stops": [
        {
            "id": "s1",
            "lat": 18.5314,
            "lon": 73.8446,
            "weather": "Sunny",
            "traffic": "Medium",
            "multiple_deliveries": 0,
        },
        {
            "id": "s2",
            "lat": 18.5074,
            "lon": 73.8077,
            "weather": "Sunny",
            "traffic": "Medium",
            "multiple_deliveries": 0,
        },
        {
            "id": "s3",
            "lat": 18.5642,
            "lon": 73.7769,
            "weather": "Sunny",
            "traffic": "Medium",
            "multiple_deliveries": 0,
        },
    ],
    "departure_time": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%dT11:00:00"),
    "algorithm": "exact",
}


def fake_matrix_builder(coordinates, departure, contexts):
    """Predictable asymmetric matrix for API contract tests."""
    n = len(coordinates)
    matrix = np.zeros((n, n), dtype=float)

    if n == 2:
        matrix[0, 1] = 10
        matrix[1, 0] = 12

    elif n == 3:
        matrix[0, 1] = 5
        matrix[1, 2] = 7
        matrix[0, 2] = 9
        matrix[2, 1] = 8
        matrix[1, 0] = 6
        matrix[2, 0] = 4

    elif n == 4:
        # Baseline route:
        # depot -> s1 -> s2 -> s3 = 30 minutes.
        matrix[0, 1] = 10
        matrix[1, 2] = 10
        matrix[2, 3] = 10

        # Other directed legs get valid positive costs.
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] == 0:
                    matrix[i, j] = 20

    return matrix


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_one_stop(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "stops": BASE_REQUEST["stops"][:1],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == ["depot", "s1"]
    assert len(body["stops"]) == 1
    assert body["total_minutes"] >= 0


def test_two_stops(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "stops": BASE_REQUEST["stops"][:2],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200

    body = response.json()

    assert body["route"][0] == "depot"
    assert set(body["route"][1:]) == {"s1", "s2"}
    assert len(body["stops"]) == 2


def test_repeated_coordinates_are_allowed(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "stops": [
            {
                "id": "s1",
                "lat": 18.5314,
                "lon": 73.8446,
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            },
            {
                "id": "s2",
                "lat": 18.5314,
                "lon": 73.8446,
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            },
        ],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200
    assert set(response.json()["route"]) == {"depot", "s1", "s2"}


def test_normal_three_stop_response_contract(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    response = client.post("/optimize", json=BASE_REQUEST)

    assert response.status_code == 200

    body = response.json()

    assert set(body) == {
        "algorithm",
        "route",
        "stops",
        "total_minutes",
        "baseline_route",
        "baseline_total_minutes",
        "improvement_pct",
        "distance_warnings",
    }

    assert body["algorithm"] == "exact"
    assert body["route"][0] == "depot"
    assert set(body["route"]) == {"depot", "s1", "s2", "s3"}
    assert len(body["stops"]) == 3

    assert body["baseline_route"] == [
        "depot",
        "s1",
        "s2",
        "s3",
    ]

    for stop in body["stops"]:
        assert {
            "id",
            "lat",
            "lon",
            "leg_minutes",
            "eta_minutes",
            "arrival_time",
        } <= set(stop)

        assert datetime.fromisoformat(stop["arrival_time"])

    assert body["total_minutes"] >= 0
    assert body["baseline_total_minutes"] == 30.0
    assert isinstance(body["improvement_pct"], (int, float))


def test_india_coordinates_are_accepted(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200


def test_duplicate_stop_ids_return_422():
    request = {
        **BASE_REQUEST,
        "stops": [
            {
                "id": "same",
                "lat": 18.5314,
                "lon": 73.8446,
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            },
            {
                "id": "same",
                "lat": 18.5074,
                "lon": 73.8077,
                "weather": "Sunny",
                "traffic": "Medium",
                "multiple_deliveries": 0,
            },
        ],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 422


def test_malformed_departure_time_returns_422():
    request = {
        **BASE_REQUEST,
        "departure_time": "not-a-date",
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 422


def test_no_stops_returns_422():
    request = {
        **BASE_REQUEST,
        "stops": [],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 422


def test_more_than_15_stops_return_422():
    stops = [
        {
            "id": f"s{i}",
            "lat": 18.50 + (i * 0.001),
            "lon": 73.80 + (i * 0.001),
            "weather": "Sunny",
            "traffic": "Medium",
            "multiple_deliveries": 0,
        }
        for i in range(1, 17)
    ]

    request = {
        **BASE_REQUEST,
        "stops": stops,
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 422


def test_exact_algorithm_is_supported(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "algorithm": "exact",
        "stops": BASE_REQUEST["stops"][:2],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200
    assert response.json()["algorithm"] == "exact"


def test_held_karp_algorithm_is_supported(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "algorithm": "held_karp",
        "stops": BASE_REQUEST["stops"][:2],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200
    assert response.json()["algorithm"] == "held_karp"


def test_hill_climbing_algorithm_is_supported(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    request = {
        **BASE_REQUEST,
        "algorithm": "hill_climbing",
        "stops": BASE_REQUEST["stops"][:2],
    }

    response = client.post("/optimize", json=request)

    assert response.status_code == 200
    assert response.json()["algorithm"] == "hill_climbing"

def test_optimize_response_includes_distance_warnings(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_eta_matrix",
        fake_matrix_builder,
    )

    response = client.post(
        "/optimize",
        json=BASE_REQUEST,
    )

    assert response.status_code == 200

    data = response.json()

    assert "distance_warnings" in data
    assert data["distance_warnings"]["count"] >= 0
    assert data["distance_warnings"]["training_range_km"] == [1.47, 20.97]
    assert "legs" in data["distance_warnings"]


# ---------------------------------------------------------------------------
# Departure Time Validation Tests (Mocked Clock)
# ---------------------------------------------------------------------------

def test_past_date_rejected(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)

    req = {**BASE_REQUEST, "departure_time": "2026-10-25T15:00:00"}
    resp = client.post("/optimize", json=req)
    assert resp.status_code == 422
    assert "departure_time cannot be in the past" in resp.text


def test_today_past_time_rejected(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    # Current time: 15:00 IST
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)

    # 15 minutes in past (outside 5-min grace window)
    req = {**BASE_REQUEST, "departure_time": "2026-10-26T14:45:00"}
    resp = client.post("/optimize", json=req)
    assert resp.status_code == 422
    assert "departure_time cannot be in the past" in resp.text


def test_today_future_time_accepted(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)
    monkeypatch.setattr("api.main.build_eta_matrix", fake_matrix_builder)

    # 1 hour in future
    req = {**BASE_REQUEST, "departure_time": "2026-10-26T16:00:00"}
    resp = client.post("/optimize", json=req)
    assert resp.status_code == 200


def test_future_date_accepted(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)
    monkeypatch.setattr("api.main.build_eta_matrix", fake_matrix_builder)

    req = {**BASE_REQUEST, "departure_time": "2026-10-30T10:00:00"}
    resp = client.post("/optimize", json=req)
    assert resp.status_code == 200


def test_timezone_handled_utc_to_ist(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    # Current time: 2026-10-26 15:00:00 IST = 2026-10-26 09:30:00 UTC
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)
    monkeypatch.setattr("api.main.build_eta_matrix", fake_matrix_builder)

    # 10:00:00 UTC = 15:30:00 IST (future in IST) -> Accepted
    req_future_utc = {**BASE_REQUEST, "departure_time": "2026-10-26T10:00:00Z"}
    resp_future = client.post("/optimize", json=req_future_utc)
    assert resp_future.status_code == 200

    # 08:00:00 UTC = 13:30:00 IST (past in IST) -> Rejected
    req_past_utc = {**BASE_REQUEST, "departure_time": "2026-10-26T08:00:00Z"}
    resp_past = client.post("/optimize", json=req_past_utc)
    assert resp_past.status_code == 422
    assert "departure_time cannot be in the past" in resp_past.text


def test_grace_period_accepted(monkeypatch):
    from zoneinfo import ZoneInfo
    ist = ZoneInfo("Asia/Kolkata")
    mock_now = datetime(2026, 10, 26, 15, 0, 0, tzinfo=ist)
    monkeypatch.setattr("api.schemas.get_current_ist_time", lambda: mock_now)
    monkeypatch.setattr("api.main.build_eta_matrix", fake_matrix_builder)

    # 3 minutes in past (within 5-min grace window)
    req = {**BASE_REQUEST, "departure_time": "2026-10-26T14:57:00"}
    resp = client.post("/optimize", json=req)
    assert resp.status_code == 200
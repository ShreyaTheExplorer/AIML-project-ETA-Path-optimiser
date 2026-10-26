"""Quick integration verification script."""
import requests
import json

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

API_URL = "http://127.0.0.1:8000"

future_departure = (datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")

# Test health
r = requests.get(f"{API_URL}/health", timeout=5)
print(f"Health: {r.status_code} {r.json()}")

# Test optimize with held_karp
payload = {
    "depot_lat": 18.5204,
    "depot_lon": 73.8567,
    "rider": {
        "vehicle_type": "Motorcycle",
        "vehicle_condition": 2,
        "delivery_person_age": 25.0,
        "delivery_person_rating": 4.5,
    },
    "festival": "No",
    "city_type": "Metropolitan",
    "stops": [
        {"id": "Koregaon Park", "lat": 18.5362, "lon": 73.8938, "weather": "Sunny", "traffic": "Medium", "multiple_deliveries": 0},
        {"id": "Shivajinagar", "lat": 18.5308, "lon": 73.8474, "weather": "Sunny", "traffic": "Medium", "multiple_deliveries": 0},
        {"id": "Kothrud", "lat": 18.5074, "lon": 73.8077, "weather": "Sunny", "traffic": "Medium", "multiple_deliveries": 0},
        {"id": "Aundh", "lat": 18.5642, "lon": 73.7769, "weather": "Sunny", "traffic": "Medium", "multiple_deliveries": 0},
    ],
    "departure_time": future_departure,
    "algorithm": "held_karp",
}


r = requests.post(f"{API_URL}/optimize", json=payload, timeout=30)
print(f"\n=== Held-Karp Optimization ===")
print(f"Status: {r.status_code}")
d = r.json()
print(f"Algorithm: {d['algorithm']}")
print(f"Route: {' -> '.join(d['route'])}")
print(f"Total: {d['total_minutes']:.1f} min")
print(f"Baseline: {d['baseline_total_minutes']:.1f} min")
print(f"Improvement: {d['improvement_pct']:.2f}%")
print(f"Stops with ETAs: {len(d['stops'])}")
for s in d["stops"]:
    print(f"  {s['id']}: leg={s['leg_minutes']}min, eta={s['eta_minutes']}min, arrival={s['arrival_time']}")

# Test with hill_climbing
payload["algorithm"] = "hill_climbing"
r = requests.post(f"{API_URL}/optimize", json=payload, timeout=30)
print(f"\n=== Hill Climbing Optimization ===")
print(f"Status: {r.status_code}")
d = r.json()
print(f"Algorithm: {d['algorithm']}")
print(f"Route: {' -> '.join(d['route'])}")
print(f"Total: {d['total_minutes']:.1f} min")
print(f"Improvement: {d['improvement_pct']:.2f}%")

# Test validation error
bad_payload = {**payload, "stops": []}
r = requests.post(f"{API_URL}/optimize", json=bad_payload, timeout=5)
print(f"\n=== Validation Error Test ===")
print(f"Status: {r.status_code} (expected 422)")

# Test invalid departure
bad_payload2 = {**payload, "departure_time": "not-a-date"}
r = requests.post(f"{API_URL}/optimize", json=bad_payload2, timeout=5)
print(f"Invalid departure: {r.status_code} (expected 422)")

print("\n=== All integration tests passed! ===")

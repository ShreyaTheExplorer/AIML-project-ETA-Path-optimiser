"""FastAPI endpoints for Delivery ETA and Route Optimization."""

from datetime import timedelta

from fastapi import FastAPI, HTTPException

from api.matrix_builder import build_eta_matrix, find_clipped_legs
from api.schemas import OptimizeRequest
from optimizer.base import route_cost
from optimizer.registry import get_solver


app = FastAPI(
    title="Delivery ETA Predictor and Route Optimizer",
    version="2.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize")
def optimize(request: OptimizeRequest):
    """
    Predict pairwise delivery ETAs and find the optimal route.
    """

    try:
        # Index 0 = depot.
        # Index 1 onward = delivery stops.
        coordinates = [
            (
                request.depot_lat,
                request.depot_lon,
            ),
            *[
                (stop.lat, stop.lon)
                for stop in request.stops
            ],
        ]

        # Validate city consistency if city_key is specified
        city_key = request.city_key.strip().lower() if request.city_key else None
        if city_key:
            from data.loader import get_city_metadata, validate_coordinates_for_city, resolve_context
            city_meta = get_city_metadata(city_key)
            depot_coords = (request.depot_lat, request.depot_lon)
            stop_coords = [(s.id, s.lat, s.lon) for s in request.stops]
            violations = validate_coordinates_for_city(city_key, depot_coords, stop_coords)
            if violations:
                raise HTTPException(
                    status_code=422,
                    detail=f"Coordinates inconsistent with city '{city_meta['display_name']}': " + "; ".join(violations),
                )
            context = resolve_context(city_key, request.departure_time)
            resolved_city_type = context["city_type"]
            resolved_festival = context["festival"] if request.festival == "No" else request.festival
        else:
            resolved_city_type = request.city_type
            resolved_festival = request.festival

        clipped_legs = find_clipped_legs(coordinates)
        # The depot does not represent a delivery destination,
        # so it does not need delivery-specific model context.
        depot_context = {}

        # Build the model context for each delivery stop.
        #
        # Route-level inputs:
        #   - rider vehicle type
        #   - rider vehicle condition
        #   - rider age
        #   - rider rating
        #   - festival
        #   - city type
        #
        # Stop-level inputs:
        #   - weather
        #   - traffic
        #   - multiple deliveries
        contexts = [
            depot_context,
            *[
                {
                    # Rider-level context
                    "vehicle_type": request.rider.vehicle_type,
                    "vehicle_condition": request.rider.vehicle_condition,
                    "delivery_person_age": request.rider.delivery_person_age,
                    "delivery_person_rating": request.rider.delivery_person_rating,

                    # Stop-level context
                    "weather": stop.weather,
                    "traffic": stop.traffic,
                    "multiple_deliveries": stop.multiple_deliveries,

                    # Route-level context
                    "festival": resolved_festival,
                    "city_type": resolved_city_type,
                }
                for stop in request.stops
            ],
        ]

        # Build pairwise ETA matrix using the trained XGBoost model.
        matrix = build_eta_matrix(
            coordinates=coordinates,
            departure=request.departure_time,
            contexts=contexts,
        )

        # Select the requested optimizer.
        solver = get_solver(request.algorithm)

        # Find the route.
        route = list(
            solver(
                matrix,
                start=0,
            )
        )

        # Baseline route:
        # depot -> stop 1 -> stop 2 -> ...
        baseline_route = list(
            range(len(coordinates))
        )

        total_minutes = float(
            route_cost(
                matrix,
                route,
            )
        )

        baseline_total_minutes = float(
            route_cost(
                matrix,
                baseline_route,
            )
        )

        if baseline_total_minutes > 0:
            improvement_pct = (
                (
                    baseline_total_minutes
                    - total_minutes
                )
                / baseline_total_minutes
                * 100
            )
        else:
            improvement_pct = 0.0

        def route_label(index: int) -> str:
            if index == 0:
                return "depot"

            return request.stops[
                index - 1
            ].id

        # Generate delivery schedule.
        departure = request.departure_time
        current_time = departure
        previous_index = route[0]

        ordered_stops = []

        for index in route[1:]:

            leg_minutes = float(
                matrix[
                    previous_index,
                    index,
                ]
            )

            current_time += timedelta(
                minutes=leg_minutes
            )

            stop = request.stops[
                index - 1
            ]

            ordered_stops.append(
                {
                    "id": stop.id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "leg_minutes": round(
                        leg_minutes,
                        2,
                    ),
                    "eta_minutes": round(
                        (
                            current_time
                            - departure
                        ).total_seconds()
                        / 60,
                        2,
                    ),
                    "arrival_time": current_time.isoformat(
                        timespec="seconds"
                    ),
                }
            )

            previous_index = index

        return {
            "algorithm": request.algorithm,
            "route": [
                route_label(index)
                for index in route
            ],
            "stops": ordered_stops,
            "total_minutes": round(
                total_minutes,
                2,
            ),
            "baseline_route": [
                route_label(index)
                for index in baseline_route
            ],
            "baseline_total_minutes": round(
                baseline_total_minutes,
                2,
            ),
            "improvement_pct": round(
                improvement_pct,
                2,
            ),
            "distance_warnings": {
                "count": len(clipped_legs),
                "training_range_km": [
                    1.47,
                    20.97,
              ],
           "legs": clipped_legs,
           },
           
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {exc}",
        ) from exc
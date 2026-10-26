from datetime import datetime, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

IST_ZONE = ZoneInfo("Asia/Kolkata")


def get_current_ist_time() -> datetime:
    """Return current datetime in Asia/Kolkata (IST). Hookable for unit tests."""
    return datetime.now(IST_ZONE)


Weather = Literal[
    "Sunny",
    "Cloudy",
    "Fog",
    "Sandstorms",
    "Stormy",
    "Windy",
    "Unknown",
]

Traffic = Literal[
    "Unknown",
    "Low",
    "Medium",
    "High",
    "Jam",
]

VehicleType = Literal[
    "Bicycle",
    "Electric Scooter",
    "Motorcycle",
    "Scooter",
]

Festival = Literal[
    "No",
    "Yes",
    "Unknown",
]

CityType = Literal[
    "Metropolitan",
    "Semi-Urban",
    "Urban",
]

Algorithm = Literal[
    "exact",
    "held_karp",
    "hill_climbing",
]


class RiderContext(BaseModel):
    """
    Information about the rider who makes all deliveries
    in the route.
    """

    model_config = ConfigDict(extra="forbid")

    vehicle_type: VehicleType = "Motorcycle"

    vehicle_condition: int = Field(
        default=2,
        ge=0,
        le=3,
    )

    delivery_person_age: float = Field(
        default=25,
        ge=20,
        le=39,
    )

    delivery_person_rating: float = Field(
        default=4.5,
        ge=2.5,
        le=5.0,
    )


class Stop(BaseModel):
    """
    One delivery location.

    Weather, traffic and order load can vary
    from one stop to another.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

    weather: Weather = "Sunny"

    traffic: Traffic = "Medium"

    multiple_deliveries: int = Field(
        default=0,
        ge=0,
        le=3,
    )

    @field_validator("id")
    @classmethod
    def validate_stop_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Stop ID cannot be empty.")

        return value


class OptimizeRequest(BaseModel):
    """
    Request sent by the frontend to optimize a delivery route.
    """

    model_config = ConfigDict(extra="forbid")

    depot_lat: float = Field(ge=-90, le=90)
    depot_lon: float = Field(ge=-180, le=180)

    rider: RiderContext = Field(
        default_factory=RiderContext
    )

    festival: Festival = "No"

    city_type: CityType = "Metropolitan"

    city_key: Optional[str] = Field(
        default=None,
        description="Optional city key (pune, mumbai, bengaluru, delhi).",
    )

    stops: list[Stop] = Field(
        min_length=1,
        max_length=15,
    )

    departure_time: datetime

    algorithm: Algorithm = "held_karp"

    @field_validator("departure_time", mode="before")
    @classmethod
    def validate_departure_time(cls, value):
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str:
                raise ValueError("departure_time cannot be empty.")
            try:
                parsed = datetime.fromisoformat(
                    value_str.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError(
                    "departure_time must be a valid ISO-format datetime."
                ) from exc
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError(
                "departure_time must be an ISO-format datetime string."
            )

        # Localize or convert to Asia/Kolkata (IST)
        if parsed.tzinfo is None:
            dt_ist = parsed.replace(tzinfo=IST_ZONE)
        else:
            dt_ist = parsed.astimezone(IST_ZONE)

        now_ist = get_current_ist_time()
        # 5-minute grace window
        if dt_ist < (now_ist - timedelta(minutes=5)):
            raise ValueError(
                f"departure_time cannot be in the past ({dt_ist.strftime('%Y-%m-%d %H:%M:%S')} IST). "
                f"Must be at or after the current time in Asia/Kolkata ({now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST, 5-minute grace window)."
            )

        return parsed

    @field_validator("stops")
    @classmethod
    def validate_unique_stop_ids(cls, stops):
        ids = [stop.id for stop in stops]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Stop IDs must be unique."
            )

        return stops
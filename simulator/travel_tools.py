import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

# Standalone default; AgentRunner overrides this from domain config data_source.
_DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "controlled_benchmark",
    "travel",
    "catalog",
)
_CONFIGURED_DB_DIR = _DEFAULT_DB_DIR


def _db_dir() -> str:
    return _CONFIGURED_DB_DIR


def configure_db_dir(path: str) -> None:
    """Select the catalog derived from the active domain data_source."""
    global _CONFIGURED_DB_DIR
    _CONFIGURED_DB_DIR = os.path.abspath(path)
    _load.cache_clear()


@lru_cache(maxsize=None)
def _load(filename: str) -> Any:
    """Load and cache a catalog file. Raises if missing."""
    path = os.path.join(_db_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog file not found: {path}. "
            "Expected <data_source>/catalog for travel."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flights() -> List[Dict[str, Any]]:
    return _load("flights.json")


def _hotels() -> List[Dict[str, Any]]:
    return _load("hotels.json")


def _policies() -> Dict[str, Any]:
    return _load("policies.json")


def search_flight(origin: str, destination: str) -> dict:
    """Search for flights between an origin and destination city.

    Matching is case-insensitive on the route endpoints.
    """
    origin_q = origin.strip().lower()
    dest_q = destination.strip().lower()
    results = []
    for flight in _flights():
        if (
            str(flight.get("origin", "")).lower() == origin_q
            and str(flight.get("destination", "")).lower() == dest_q
        ):
            results.append(
                {
                    "id": flight["id"],
                    "airline": flight.get("airline"),
                    "origin": flight.get("origin"),
                    "destination": flight.get("destination"),
                    "departure": flight.get("departure"),
                    "arrival": flight.get("arrival"),
                    "cabin": flight.get("cabin"),
                    "price": flight.get("price"),
                    "seats_available": flight.get("seats_available"),
                }
            )

    return {
        "origin": origin,
        "destination": destination,
        "results": results,
        "count": len(results),
    }


def search_hotel(destination: str) -> dict:
    """Search for hotels in a destination city.

    Matching is a case-insensitive token overlap against the hotel's city,
    name, and amenities.
    """
    tokens = [t for t in destination.lower().split() if t]
    results = []
    for hotel in _hotels():
        haystack = " ".join(
            str(hotel.get(field, ""))
            for field in ("city", "name", "amenities", "neighborhood")
        ).lower()
        if not tokens or any(token in haystack for token in tokens):
            results.append(
                {
                    "id": hotel["id"],
                    "name": hotel["name"],
                    "city": hotel.get("city"),
                    "stars": hotel.get("stars"),
                    "price_per_night": hotel.get("price_per_night"),
                    "rooms_available": hotel.get("rooms_available"),
                }
            )

    return {
        "destination": destination,
        "results": results,
        "count": len(results),
    }


def check_policy(airline: str, policy_type: Optional[str] = None) -> dict:
    """Check an airline's travel policies (baggage, cancellation, change fees, visa).

    Matching on the airline name is case-insensitive. When ``policy_type`` is
    given, only that policy is returned; otherwise the full policy set for the
    airline is returned.
    """
    policies = _policies()
    airline_q = airline.strip().lower()
    entry = None
    for name, value in policies.items():
        if name.lower() == airline_q:
            entry = value
            break

    if entry is None:
        return {"airline": airline, "error": "airline not found in policy table"}

    if policy_type is None:
        return {"airline": airline, "policies": entry}

    key = policy_type.strip().lower()
    row = entry.get(key)
    if row is None:
        return {
            "airline": airline,
            "policy_type": policy_type,
            "note": "no policy on file for this policy type",
        }

    return {
        "airline": airline,
        "policy_type": policy_type,
        "policy": row,
    }

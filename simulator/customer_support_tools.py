import json
import os
import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional

# Standalone default; AgentRunner overrides this from domain config data_source.
_DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "controlled_benchmark",
    "customer_support",
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
            "Expected <data_source>/catalog for customer_support."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _orders() -> Dict[str, Any]:
    return _load("orders.json")


def _refund_policies() -> Dict[str, Any]:
    return _load("refund_policies.json")


def lookup_order(order_id: str) -> dict:
    """Look up an order's status, items, and shipping state by order ID."""
    orders = _orders()
    entry = orders.get(order_id)
    if entry is None:
        return {"order_id": order_id, "error": "order not found"}

    return {
        "order_id": order_id,
        "status": entry.get("status"),
        "items": entry.get("items", []),
        "placed_at": entry.get("placed_at"),
        "estimated_delivery": entry.get("estimated_delivery"),
        "total": entry.get("total"),
    }


def create_ticket(
    subject: str,
    order_id: Optional[str] = None,
    priority: str = "medium",
) -> dict:
    """Open a new support ticket for an issue, optionally tied to an order.

    A deterministic ticket ID is derived from the subject and order so the same
    request reproduces the same ticket. If ``order_id`` is given but unknown,
    the ticket is still created and flagged so the caller can correct it.
    """
    valid_priorities = ("low", "medium", "high", "urgent")
    priority = priority.strip().lower()
    if priority not in valid_priorities:
        priority = "medium"

    order_known = order_id is None or order_id in _orders()
    ticket_seed = f"{subject.strip().lower()}|{order_id or ''}".encode("utf-8")
    seed = int(hashlib.sha256(ticket_seed).hexdigest()[:12], 16) % 100000
    ticket_id = f"TCK{seed:05d}"

    result = {
        "ticket_id": ticket_id,
        "status": "open",
        "subject": subject,
        "priority": priority,
        "order_id": order_id,
        "created_at": "2026-06-15T00:00:00",
    }
    if not order_known:
        result["warning"] = f"order {order_id} not found; ticket created without a linked order"
    return result


def check_refund_policy(category: str) -> dict:
    """Look up the refund policy for a product category.

    Matching on the category name is case-insensitive. Falls back to the
    ``default`` policy when the category has no specific entry.
    """
    policies = _refund_policies()
    category_q = category.strip().lower()
    for name, value in policies.items():
        if name.lower() == category_q:
            return {"category": category, "policy": value}

    default = policies.get("default")
    if default is not None:
        return {"category": category, "policy": default, "note": "no category-specific policy; default applied"}

    return {"category": category, "error": "no refund policy on file"}

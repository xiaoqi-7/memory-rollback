import os
from typing import Callable, Dict

from simulator.shopping_tools import (
    compare_price,
    check_shipping,
    configure_db_dir as configure_shopping_db_dir,
    search_product,
)
from simulator.customer_support_tools import (
    configure_db_dir as configure_customer_support_db_dir,
    create_ticket,
    lookup_order,
    check_refund_policy,
)
from simulator.travel_tools import (
    check_policy,
    configure_db_dir as configure_travel_db_dir,
    search_flight,
    search_hotel,
)
from simulator.lme_derived_tools import (
    configure_db_dir as configure_lme_derived_db_dir,
    lookup_state,
    lookup_trajectory,
    search_trajectory_history,
)


_CATALOG_CONFIGURERS = {
    "shopping": configure_shopping_db_dir,
    "customer_support": configure_customer_support_db_dir,
    "travel": configure_travel_db_dir,
    "lme_derived": configure_lme_derived_db_dir,
}


def configure_data_source(domain: str, data_source: str) -> None:
    """Derive the active simulator catalog from a domain config data_source."""
    configure = _CATALOG_CONFIGURERS.get(domain)
    if configure is None:
        raise ValueError(f"No catalog configuration for domain {domain!r}")
    configure(os.path.join(data_source, "catalog"))

TOOLS: Dict[str, Dict[str, Callable]] = {
    "shopping": {
        "search_product": search_product,
        "compare_price": compare_price,
        "check_shipping": check_shipping,
    },
    "customer_support": {
        "lookup_order": lookup_order,
        "create_ticket": create_ticket,
        "check_refund_policy": check_refund_policy,
    },
    "travel": {
        "search_flight": search_flight,
        "search_hotel": search_hotel,
        "check_policy": check_policy,
    },
    "lme_derived": {
        "search_trajectory_history": search_trajectory_history,
        "lookup_state": lookup_state,
        "lookup_trajectory": lookup_trajectory,
    },
}

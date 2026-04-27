"""
Service registry (Phase 2, Step 3).

A single, cached entry point that wires every Phase 2 service to the same
SQLAlchemy engine.  Downstream code (the FastAPI layer in Phase 5, the
Streamlit dashboard in Phase 6) only needs to call :func:`get_registry`;
each service instance is created once per engine and reused thereafter.

Usage
-----
>>> from app.models.schema import get_engine
>>> from app.services.service_registry import get_registry
>>> reg = get_registry(get_engine())
>>> reg.kpi.get_lot_risk_scores().head()
>>> reg.drill_down.get_full_drill_down_chain("L-778")

Cache semantics
---------------
The cache key is :func:`id(engine)` -- i.e. each distinct Engine *object*
(not connection string) gets its own registry.  Rebuilding the engine
(for tests, or a pytest fixture with an in-memory DB) returns a fresh
registry.  Call :func:`clear_registry_cache` to force reinitialisation.
"""
from __future__ import annotations

import logging
from typing import Dict

from sqlalchemy.engine import Engine

from app.services.anomaly_detector import AnomalyDetector
from app.services.drill_down import DrillDownService
from app.services.explainer import InsightExplainer
from app.services.kpi_engine import KPIEngine
from app.services.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """Bundle of every Phase 2 service, all bound to a single engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

        # Cheap services first -- no DB work at init time.
        self.kpi = KPIEngine(engine)
        self.anomaly = AnomalyDetector(engine)
        self.drill_down = DrillDownService(engine)

        # These read reference tables at init, so build them last so any
        # failures hit after the cheaper services are already queryable.
        self.explainer = InsightExplainer(engine)
        self.recommendations = RecommendationEngine(engine)

        logger.debug("ServiceRegistry initialised with engine=%r", engine)

    def all_services(self) -> Dict[str, object]:
        """Return a name->instance map -- handy for introspection + tests."""
        return {
            "kpi":             self.kpi,
            "anomaly":         self.anomaly,
            "explainer":       self.explainer,
            "drill_down":      self.drill_down,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------

# Keyed on id(engine): SQLAlchemy Engine objects are rebuilt per
# process / test session, so identity is a safe cache key.
_REGISTRY_CACHE: Dict[int, ServiceRegistry] = {}


def get_registry(engine: Engine) -> ServiceRegistry:
    """Return a :class:`ServiceRegistry` for *engine*, building it once.

    Subsequent calls with the same engine object return the same registry;
    a different engine (or the same engine after :func:`clear_registry_cache`)
    rebuilds from scratch.
    """
    key = id(engine)
    if key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[key] = ServiceRegistry(engine)
        logger.info("ServiceRegistry built for engine id=%s", key)
    return _REGISTRY_CACHE[key]


def clear_registry_cache() -> None:
    """Drop every cached registry -- used by test fixtures."""
    _REGISTRY_CACHE.clear()


__all__ = ["ServiceRegistry", "get_registry", "clear_registry_cache"]

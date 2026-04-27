"""
Response caching for the Quality Agent (Phase 5 Step 2).

A small, thread-safe TTL cache keyed by the question string. Used both by
the live agent and the demo-mode mock responder so that the **first time**
a judge clicks one of the suggested questions the response comes back
under 100 ms — not the 5–8 s a cold tool-use loop would normally take.

Two SLA tiers we instrument and warn on:

  * cache hit:                 < 100  ms
  * database-only path:        < 500  ms
  * full agent (Claude API):   < 8000 ms

Anything over 8000 ms emits a WARNING-level log.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300        # 5 minutes
DEFAULT_MAX_SIZE    = 64
SLA_WARN_MS         = 8000.0     # full-agent slow-call threshold
SLA_DB_MS           = 500.0      # database-only target
SLA_CACHE_MS        = 100.0      # cache-hit target


# ---------------------------------------------------------------------------
# QueryCache
# ---------------------------------------------------------------------------

class QueryCache:
    """Thread-safe LRU + TTL cache for response objects.

    Keys are SHA-256 hashes of the lower-cased, stripped question string,
    so semantically identical questions ("L-778 risk?" vs "  L-778 risk? ")
    share the same cache entry.

    Values are arbitrary objects (typically AgentResponse or the dict
    returned by `render_mock_response`).
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_size:    int = DEFAULT_MAX_SIZE,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size    = max_size
        self._store: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()
        self._hits   = 0
        self._misses = 0

    # ── Key normalisation ─────────────────────────────────────────────────
    @staticmethod
    def make_key(question: str) -> str:
        normalised = (question or "").lower().strip()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────
    def get(self, question: str) -> Optional[Any]:
        key = self.make_key(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if (time.time() - ts) > self.ttl_seconds:
                # Expired — drop it.
                self._store.pop(key, None)
                self._misses += 1
                return None
            # Refresh ordering for crude LRU behaviour.
            self._store.pop(key, None)
            self._store[key] = (ts, value)
            self._hits += 1
            return value

    def set(self, question: str, value: Any) -> None:
        key = self.make_key(question)
        with self._lock:
            if key in self._store:
                self._store.pop(key, None)
            elif len(self._store) >= self.max_size:
                # Evict the oldest by insertion order (Python dict preserves it).
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), value)

    def invalidate(self, question: str) -> bool:
        key = self.make_key(question)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "size":      len(self._store),
                "max_size":  self.max_size,
                "ttl_s":     self.ttl_seconds,
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  round(hit_rate, 3),
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default: Optional[QueryCache] = None
_default_lock = threading.Lock()


def get_default_cache() -> QueryCache:
    global _default
    with _default_lock:
        if _default is None:
            _default = QueryCache()
        return _default


def reset_default_cache() -> None:
    """Test-only — drop the singleton so each suite starts clean."""
    global _default
    with _default_lock:
        _default = None


# ---------------------------------------------------------------------------
# SLA-aware timing helpers
# ---------------------------------------------------------------------------

def annotate_response(
    response: Any,
    response_time_ms: float,
    cache_hit: bool,
) -> Any:
    """Attach response_time_ms / cache_hit to the response if it's mutable.

    Works for both dict-shaped mock responses and dataclass-shaped
    AgentResponse instances. Returns the same object (mutated in place).
    """
    if response is None:
        return response
    try:
        if isinstance(response, dict):
            response["response_time_ms"] = round(response_time_ms, 2)
            response["cache_hit"]        = bool(cache_hit)
        else:
            setattr(response, "response_time_ms", round(response_time_ms, 2))
            setattr(response, "cache_hit",        bool(cache_hit))
    except Exception:  # noqa: BLE001 — instrumentation must never crash callers
        pass

    if response_time_ms > SLA_WARN_MS:
        logger.warning(
            "SLA breach: response took %.0f ms (target <= %.0f ms)",
            response_time_ms, SLA_WARN_MS,
        )
    return response


def cached_call(
    cache: QueryCache,
    question: str,
    producer,
):
    """Run *producer()* with cache-aware timing + SLA annotation.

    Returns ``(value, response_time_ms, cache_hit)``.
    """
    start = time.perf_counter()
    cached = cache.get(question)
    if cached is not None:
        elapsed = (time.perf_counter() - start) * 1000
        annotate_response(cached, elapsed, cache_hit=True)
        return cached, elapsed, True

    value = producer()
    elapsed = (time.perf_counter() - start) * 1000
    cache.set(question, value)
    annotate_response(value, elapsed, cache_hit=False)
    return value, elapsed, False


# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------

DEFAULT_WARM_QUESTIONS: List[str] = [
    "What is the risk level of lot L-778 and what actions should I take?",
    "Are there any process drift issues on our production lines?",
    "Compare SUP-A and SUP-B for a new safety-critical program",
    "Why did serial SR20260008 fail in the field?",
    "Where should I focus incoming inspection effort this week?",
]


def pre_warm_cache(
    *,
    agent=None,
    registry=None,
    engine=None,
    questions: Optional[List[str]] = None,
    cache: Optional[QueryCache]    = None,
) -> Dict[str, Any]:
    """Pre-warm the response cache with the demo questions.

    If `agent` is given, calls `agent.ask(q)` for each (live LLM path).
    Otherwise falls back to `render_mock_response(q, registry, engine)`,
    which is the fast offline path used in demo mode.

    Returns a small report dict so callers can log how warming went.
    """
    cache = cache or get_default_cache()

    if questions is None:
        if agent is not None and hasattr(agent, "get_suggested_questions"):
            try:
                questions = agent.get_suggested_questions()
            except Exception:
                questions = DEFAULT_WARM_QUESTIONS
        if not questions:
            questions = DEFAULT_WARM_QUESTIONS

    if agent is None and (registry is None or engine is None):
        raise ValueError(
            "pre_warm_cache requires either `agent` or both `registry` + `engine`"
        )

    started = time.perf_counter()
    timings: List[float] = []
    errors:  List[str]   = []

    for q in questions:
        t0 = time.perf_counter()
        try:
            if agent is not None:
                value = agent.ask(q)
            else:
                from app.agent.mock_responder import render_mock_response
                value = render_mock_response(q, registry, engine)
            elapsed = (time.perf_counter() - t0) * 1000
            cache.set(q, value)
            annotate_response(value, elapsed, cache_hit=False)
            timings.append(elapsed)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q!r}: {exc!r}")
            logger.warning("pre-warm failed for %r: %s", q, exc)

    total_ms = (time.perf_counter() - started) * 1000
    report = {
        "warmed":      len(timings),
        "errors":      errors,
        "total_ms":    round(total_ms, 2),
        "avg_ms":      round(sum(timings) / len(timings), 2) if timings else 0.0,
        "questions":   list(questions),
        "cache_stats": cache.stats(),
    }
    logger.info(
        "Cache pre-warmed with %d response(s) in %.0f ms",
        report["warmed"], report["total_ms"],
    )
    return report


__all__ = [
    "QueryCache",
    "annotate_response",
    "cached_call",
    "get_default_cache",
    "pre_warm_cache",
    "reset_default_cache",
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_WARM_QUESTIONS",
    "SLA_CACHE_MS",
    "SLA_DB_MS",
    "SLA_WARN_MS",
]

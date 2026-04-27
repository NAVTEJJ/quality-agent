# Technical Highlights — AI Quality Inspection Copilot

---

## Architecture

```
Excel data → Ingestion pipeline → SQLite star schema (14 tables)
                                        │
                              KPIService / DrillDownService
                              (SQLAlchemy Core, parameterised queries)
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
             FastAPI layer        Claude Agent          Streamlit UI
             (REST endpoints)     (tool use, Claude     (4 screens)
                                   Sonnet 4.5)
                                        │
                              QueryCache (TTL=300s)
                              + DrillDownService._CHAIN_CACHE
```

**Data model:** Star schema with 5 dimension tables (`dim_lot`, `dim_supplier`, `dim_component`, `dim_serial`, `dim_material`), 4 fact tables (`fact_incoming_qm`, `fact_process_measurements`, `fact_warranty_claims`, `fact_constituent_bom`), 4 aggregation tables, and 2 reference tables (`ref_action_playbook`, `ref_ai_insights`).

---

## What Differentiates This from a RAG Chatbot

| Capability | RAG chatbot | This system |
|---|---|---|
| Data source | Embedded document chunks | Live SQL queries against star schema |
| Number accuracy | Approximate (from text) | Exact (from database rows) |
| Drill-down | Flat document retrieval | Graph traversal: lot → serial → process → warranty |
| Traceability | "I found this in document X" | Every number traceable to a specific table and row |
| Action output | General advice | Curated playbook from `ref_action_playbook` |
| Latency (cache hit) | 200–2000 ms (embedding search) | < 100 ms (in-memory cache) |

The system uses Claude for reasoning and language generation only. All data retrieval is via parameterised SQL — the language model never touches raw numbers.

---

## Key Technical Decisions

### 1. Star Schema Over Document Store
Quality data has natural relational structure. A star schema supports precise joins (e.g., "warranty claims for serials built from lot L-778") that document retrieval cannot answer without pre-computing every possible combination.

### 2. Two-Layer Cache
- **`QueryCache`** (TTL 300s, max 64 entries, SHA-256 key): covers repeated agent questions during a demo session. Thread-safe via `threading.RLock`.
- **`DrillDownService._CHAIN_CACHE`** (process-lifetime): the drill-down fan-out (55 serials × 3 queries each) takes ~500 ms cold. The class-level dict reduces subsequent calls to < 5 ms.

### 3. Demo Mode Without API Key
`render_mock_response()` in `app/agent/mock_responder.py` implements the full routing logic (14 branches, entity extraction via regex) using the same KPI service methods as the live agent. Judges see real data with realistic response formatting even if `ANTHROPIC_API_KEY` is absent.

### 4. Adversarial Hardening
`tests/test_phase5_hardening.py` contains 32 test cases across 6 groups:
- Ambiguous questions (must not crash, must return substantive content)
- Invalid entity lookups (must return "not found" with suggestions, not a traceback)
- Follow-up without context (must ask for clarification, not hallucinate a lot)
- Concurrent stress (5 threads simultaneously — audit log verified)
- SQL injection / special characters (parameterised queries verified safe)
- 12 exact spec questions with content-check assertions

### 5. Deterministic Risk Scoring
`KPIService.get_lot_risk_scores()` computes a weighted composite score from three independent signals:
- Incoming quality fail rate (weight 0.4)
- Warranty claim rate (weight 0.4)
- Supplier process Cpk inverted (weight 0.2)

Thresholds are fixed constants, not ML-derived. Every decision is auditable.

---

## Metrics

| Metric | Value |
|---|---|
| Test pass rate | 32 / 32 (100%) |
| Mock response latency (cache hit) | < 100 ms |
| Mock response latency (cold) | < 300 ms |
| DB query p95 — lot risk scores | < 200 ms |
| DB query p95 — process drift | < 100 ms |
| Drill-down chain (cached) | < 5 ms |
| Drill-down chain (cold) | < 500 ms |
| Streamlit screen load p95 | < 1000 ms |
| Pre-demo check wall-clock | < 10 s |
| Database tables | 14 |
| Lots in warehouse | 200 |
| Inspection records | 1,000+ |
| Warranty claims | 13 |

---

## Files of Note

| File | Purpose |
|---|---|
| `app/core/cache.py` | QueryCache, pre_warm_cache, annotate_response |
| `app/agent/mock_responder.py` | 14-branch mock responder with entity extraction |
| `app/agent/agent_core.py` | Claude agent with cache fast-path |
| `app/services/kpi_service.py` | All KPI computations (SQL → DataFrame) |
| `app/services/drill_down.py` | Graph traversal with in-memory chain cache |
| `app/agent/api.py` | FastAPI: /health, /lot/{lot}/risk, /process-drift, /supplier/{id}/profile |
| `scripts/pre_demo_check.py` | 8-step pre-demo validator (< 10 s wall-clock) |
| `scripts/reliability_check.py` | 5 health checks returning PASS/WARN/FAIL |
| `scripts/performance_test.py` | p50/p95/p99 latency suite (DB + API + Streamlit) |
| `tests/test_phase5_hardening.py` | 32 adversarial test cases (6 groups) |

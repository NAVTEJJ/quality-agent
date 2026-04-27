# AI Quality Inspection Copilot
> Industry-grade AI agent for manufacturing quality management.
> Built with Claude AI · FastAPI · Streamlit · SQLite

---

## What It Does

Transforms raw manufacturing quality data into actionable intelligence — surfacing risky lots, detecting process drift, profiling suppliers, and tracing field failures back to their root cause. All connected to your SAP QM workflow. Ask a plain-English question and get a grounded, cited answer in under a second.

---

## Live Demo

**5 questions to run through on demo day:**

| # | Question | Expected Output |
|---|---|---|
| 1 | `What is the risk level of lot L-778 and what actions should I take?` | HIGH risk · score 0.847 · 8 warranty claims · SAP QA32/QM01 actions |
| 2 | `Which production lines are showing abnormal torque failure rates by shift?` | LINE-2 Night flagged at 23.1% — 4x plant average |
| 3 | `Compare SUP-B against other suppliers and tell me if they are worth a premium contract.` | SUP-B scores 89 · Preferred tier · beats Japan COO benchmark |
| 4 | `Why did serial SR20260008 fail in the field?` | Lot → component → process measurement → warranty claim chain |
| 5 | `Where should I focus inspection this week?` | Risk-based priority list: block / 100% inspect / watchlist |

---

## Quick Start

```bash
git clone <repo-url>
cd quality-agent

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env (optional — full demo mode works without it)

python main.py          # ingest data, compute KPIs, validate all demo stories

streamlit run app/frontend/streamlit_app.py
# Open http://localhost:8501
```

**Windows one-liner:** `run.bat`
**Mac / Linux:** `bash run.sh`

---

## Architecture

```
Excel workbook (6 tabs, 15K+ rows)
        │
        ▼  NormalizationPipeline
┌──────────────────────────────────────────────────────┐
│  Phase 1 — Data Foundation                           │
│  14 tables · 4,341 rows · SQLite star schema         │
│  dim_supplier · dim_component · dim_lot · dim_serial  │
│  fact_incoming_qm · fact_process_measurements         │
│  fact_warranty_claims · fact_constituent_bom          │
│  agg_supplier_scorecard · agg_coo_trends · ...       │
└───────────────────┬──────────────────────────────────┘
                    │ SQLAlchemy Core
        ▼  KPIService / DrillDownService / RecommendationService
┌──────────────────────────────────────────────────────┐
│  Phase 2 — Insight Engine                            │
│  Lot risk scoring (4-signal weighted formula)        │
│  Process drift detection (by line + shift)           │
│  Supplier intelligence + COO decomposition           │
│  Full traceability: lot → serial → warranty          │
└───────────────────┬──────────────────────────────────┘
                    │ Tool use
        ▼  QualityAgent (Claude claude-sonnet-4-6)
┌──────────────────────────────────────────────────────┐
│  Phase 3 — Claude Agent                              │
│  10 tools · 10 intents · conversation memory         │
│  Entity resolution · audit log (JSONL)               │
│  FastAPI: /health · /lot/:id/risk · /process-drift   │
└───────────────────┬──────────────────────────────────┘
                    │ st.cache_resource
        ▼  Streamlit (4 screens)
┌──────────────────────────────────────────────────────┐
│  Phase 4 — Frontend                                  │
│  Screen A: AI Copilot (chat + context panel)         │
│  Screen B: Quality Dashboard (KPI cards + charts)    │
│  Screen C: Lot Drill-Down (6-level evidence chain)   │
│  Screen D: Analytics (Plotly charts + export)        │
└───────────────────┬──────────────────────────────────┘
                    │ cache.py
┌──────────────────────────────────────────────────────┐
│  Phase 5 — Demo Hardened                             │
│  QueryCache (TTL 300s) · drill-down chain cache      │
│  pre_warm_cache() on startup (<100ms suggested qs)   │
│  scripts/pre_demo_check.py (<10s full validation)    │
│  32-case adversarial test suite (100% passing)       │
└──────────────────────────────────────────────────────┘
```

---

## Key Capabilities

- **Lot risk scoring** — weighted 4-signal formula: fail rate · warranty rate · supplier Cpk · COO index
- **Process drift detection** — flagged by production line and shift; LINE-2 Night at 23.1%
- **Supplier intelligence** — tier-stratified scorecards with COO decomposition and premium fit analysis
- **Full traceability** — lot → incoming inspection → affected serials → process measurements → warranty claims
- **SAP-native recommendations** — every response includes specific SAP transaction codes (QA32, QE51N, QM01)
- **Conversation memory** — entity resolution across turns; context panel tracks current lot + supplier
- **Complete audit trail** — every agent query logged to `data/processed/audit_log.jsonl`
- **Demo mode** — full UI with real data and grounded AI responses, no API key required

---

## SAP Integration Points

| Transaction | When It Triggers |
|---|---|
| **QA32** | Review incoming inspection lot for a flagged risk |
| **QE51N** | Update inspection plan after drift detection |
| **QM01** | Create quality notification for HIGH-risk supplier |
| **ME57** | Source list review when recommending supplier switch |
| **MB51** | Material movement trace for affected serials |

---

## Test Coverage

```
tests/test_phase1.py     — data ingestion + schema integrity
tests/test_phase2.py     — KPI engine + demo story assertions
tests/test_phase3.py     — agent tools + intent classification
tests/test_phase4.py     — frontend imports + UI component load
tests/test_phase5_hardening.py  — 32 adversarial test cases
                                  (ambiguous qs · invalid entities · SQL injection ·
                                   concurrent stress · 12 exact spec questions)
```

Run all: `python -m pytest tests/ -v`

---

## Demo Hardening

```bash
# 60-second pre-flight check — run this 5 minutes before presenting
python scripts/pre_demo_check.py

# Performance benchmarks (p50/p95/p99 latency)
python scripts/performance_test.py

# Individual health checks
python scripts/reliability_check.py
```

**SLA targets:**
- Suggested question (cache hit): < 100 ms
- Full DB query: < 200 ms p95
- Drill-down chain (cached): < 5 ms

---

## Project Structure

```
quality-agent/
├── app/
│   ├── agent/              # QualityAgent, API, mock_responder, tool definitions
│   ├── core/               # cache.py — QueryCache + pre_warm_cache
│   ├── frontend/
│   │   ├── components/     # screen_a through screen_d, kpi_cards, export
│   │   ├── theme.py        # dark CSS, color palette
│   │   └── streamlit_app.py
│   ├── ingestion/          # Excel loader, normalizer, data profiler
│   ├── models/             # SQLAlchemy schema (14 tables)
│   └── services/           # KPIService, DrillDownService, RecommendationService, …
├── configs/settings.py
├── data/
│   ├── raw/                # original Excel source file
│   └── processed/          # quality.db + insights.json + audit_log.jsonl
├── docs/
│   ├── DEMO_SCRIPT.md      # 5-minute demo flow + judge Q&A
│   └── TECHNICAL_HIGHLIGHTS.md
├── scripts/
│   ├── pre_demo_check.py   # 8-step pre-demo validator
│   ├── reliability_check.py
│   └── performance_test.py
├── tests/                  # 66+ tests across 5 phases
├── main.py                 # Phase runners + validation
├── run.bat                 # Windows one-click launcher
├── run.sh                  # Mac/Linux one-click launcher
└── requirements.txt
```

---

## Running Individual Phases

```bash
python main.py phase1          # Ingest Excel → SQLite (14 tables)
python main.py phase2          # KPI engine + insight generation
python main.py phase3          # Agent boot + 5 demo questions + FastAPI
python main.py phase4          # Frontend validation + Streamlit launch
python main.py full            # All phases end-to-end
```

---

_Built with Claude claude-sonnet-4-6 · Streamlit · SQLAlchemy · Plotly · FastAPI · Python 3.11+_

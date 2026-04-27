# Demo Script — AI Quality Inspection Copilot
**5-minute hackathon demo · Phase 5**

---

## Opening (30 seconds)

> "Every manufacturer drowns in quality data but can't act on it fast enough.
> This is an AI copilot that sits on top of a real star-schema warehouse,
> understands the full evidence chain from incoming inspection through
> process measurements to warranty outcomes, and gives a quality engineer
> a decisive answer in under a second."

Open Streamlit. Point to the sidebar alert:
> "The system already flagged something — lot L-778 is HIGH risk.
> Let's find out why."

---

## Demo Question 1 — Lot Risk

**Type into the chat box:**
```
What is the risk level of lot L-778 and what actions should I take?
```

**Talking points while the response loads:**
- Response returns from cache in < 100 ms (pre-warmed on startup)
- Risk tier is derived from a composite of incoming fail rate, process Cpk, and warranty claim rate — not a single threshold
- The action playbook is pulled from `ref_action_playbook` — a curated table of engineer-approved interventions, not hallucinated text

---

## Demo Question 2 — Process Drift

**Type:**
```
Which production lines are showing abnormal torque failure rates by shift?
```

**Talking points:**
- LINE-2 Night is highlighted at 23.1% torque fail rate — more than double the plant average
- This surfaces a shift-specific tooling or operator pattern that aggregate dashboards would miss
- The agent cross-references `fact_process_measurements` with shift metadata — it's not just counting failures, it's localising them

---

## Demo Question 3 — Supplier Intelligence

**Type:**
```
Compare SUP-B against other suppliers and tell me if they are worth a premium contract.
```

**Talking points:**
- SUP-B scores 89 quality points — top-ranked, Preferred tier
- The COO decomposition shows SUP-B is at parity with the Japan benchmark (their country of origin), meaning their quality is structural, not lucky
- The recommendation is grounded in on-time delivery, warranty claim rate, and process Cpk — the same criteria a sourcing manager would use

---

## Demo Question 4 — Drill-Down Chain

**Switch to the Lot Drill-Down screen (Screen C). Select L-778.**

**Talking points:**
- One click traverses the full evidence chain: lot → inspection records → affected serials → process measurements → warranty outcomes → supplier scorecard → COO benchmark
- 8 warranty claims on L-778 serials — the system links incoming quality defects to field failures, closing the feedback loop that most QMS systems leave open
- This is a graph traversal over a star schema, not a document search — every number is traceable to a row in the database

---

## Demo Question 5 — Safety-Critical Risk

**Switch back to AI Copilot. Type:**
```
Are there any safety-critical components at risk?
```

**Talking points:**
- The agent filters on `safety_critical = 1` in `dim_component` before scoring risk
- This is the question a chief engineer asks on a Friday afternoon — the system answers it in one shot rather than requiring three separate pivot tables
- The response includes the component name, the lot, and the recommended hold action

---

## Closing Line

> "In five questions we went from a raw quality alert to a sourcing recommendation backed by warranty data.
> A quality engineer running this manually would need three systems and half a day.
> The copilot does it in under five seconds, with every answer traceable to the source row."

---

## Judge Q&A — Prepared Answers

### Q1: "How does the AI decide what's HIGH risk?"

The risk tier is computed in `KPIService.get_lot_risk_scores()` using a weighted composite:
- Incoming fail rate from `fact_incoming_qm`
- Warranty claim rate from `fact_warranty_claims` joined via `dim_serial`
- Process Cpk from `dim_supplier.process_cpk`

Thresholds: HIGH ≥ 0.60 composite score, MEDIUM ≥ 0.35, LOW below that.
No ML model — the scoring is deterministic and auditable.

---

### Q2: "What stops the AI from hallucinating numbers?"

Every number in every response comes from a SQL query, not from the language model.
The agent uses Claude only for reasoning and language — data retrieval is always via parameterised queries against the SQLite warehouse.
The mock responder (demo mode) follows the same pattern: it calls the same KPI service methods and formats the results.

---

### Q3: "Could this scale to a real production database?"

The schema is standard star-schema SQL — swap `sqlite:///` for `postgresql://` in `.env` and nothing else changes.
The KPI service uses SQLAlchemy Core so it's dialect-agnostic.
The drill-down chain cache would need Redis instead of a class-level dict for multi-worker deployments, but the interface is identical.

---

### Q4: "How long did this take to build?"

Five phases over the hackathon window:
1. Schema design and data ingestion pipeline
2. KPI service layer and drill-down graph traversal
3. FastAPI + Streamlit frontend
4. Claude agent integration with tool use
5. Adversarial hardening, performance optimisation, and reliability checks

The adversarial test suite (32 test cases) and performance harness were built in Phase 5 specifically to make the demo bulletproof.

---

### Q5: "What's the most technically interesting part?"

The drill-down chain traversal. A single `get_full_drill_down_chain('L-778')` call:
- Joins 6 tables
- Fans out to 55 affected serials
- Fetches process measurements and warranty outcomes per serial
- Computes a COO benchmark gap
- Returns a fully JSON-serialisable evidence tree in < 500 ms cold, < 5 ms cached

The in-memory cache uses a class-level dict keyed on normalised lot number — judges can click L-778 repeatedly with zero latency.

---

### Q6: "Why Streamlit instead of React?"

Streamlit lets a single engineer ship a production-quality data UI in hours rather than days.
For a hackathon judged on insight depth rather than frontend polish, that tradeoff is correct.
The architecture separates concerns cleanly enough that the Streamlit layer could be replaced with a React frontend consuming the FastAPI endpoints without touching any business logic.

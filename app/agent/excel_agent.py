"""
ExcelQualityAgent -- code-execution architecture.

Claude receives the Excel schema + known facts, writes Python/pandas code to
answer each question, executes against the actual .xlsx file in real time, then
formats the final answer with sheet+column citations.

Flow per question:
  1. User question → Claude (with cached system prompt)
  2. Claude calls run_python with pandas code
  3. Code executes; stdout returned as tool_result
  4. Claude reads output → writes final answer (plain prose, cited numbers)
  5. Loop repeats up to MAX_TOOL_ROUNDS if Claude needs a follow-up query
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import pandas as pd

from configs import settings

logger = logging.getLogger(__name__)

EXCEL_PATH: Path = settings.EXCEL_PATH
MAX_TOOL_ROUNDS = 6
EXEC_TIMEOUT_S = 30

try:
    EXCEL_CACHE: Optional[pd.ExcelFile] = pd.ExcelFile(str(EXCEL_PATH))
except Exception:
    EXCEL_CACHE = None

# ── System prompt (cached after first call) ───────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert manufacturing quality engineer with full Python/pandas access
to the production quality workbook.

CAPABILITY
----------
Write Python code to answer every question by querying the actual Excel file.
EXCEL_PATH (a Path object), pd (pandas), and EXCEL_CACHE (a pre-loaded pd.ExcelFile)
are all pre-loaded. Do NOT write import pandas. Do NOT hardcode any path.
Load sheets faster using the pre-loaded ExcelFile:
    df = pd.read_excel(EXCEL_CACHE, sheet_name="SheetName")
Always print() results so they appear in the tool output.

SCHEMA
------
AsBuilt_Serial (300 rows)
  SerialNo*, FinishedMaterial, Plant, Line, Shift,
  VendorOfCriticalAssy, COO, ECN_Level
  (* primary key linking all operational sheets)

Constituent_BOM (1200 rows = 4 components × 300 serials)
  SerialNo, Component, Supplier, COO, LotNo, MfgDate, CertDocID

Incoming_QM (1486 rows)
  Component, Supplier, LotNo, Result (values: PASS or FAIL), DefectCode
  NOT keyed by SerialNo — join to Constituent_BOM on (Component, Supplier, LotNo)
  Compute fail rate as: (df["Result"] == "FAIL").sum() / len(df)

Process_Measurements (300 rows)
  SerialNo, Line, Shift, Torque_Nm, Torque_Result (PASS/FAIL),
  LeakRate_ccm, Leak_Result (PASS/FAIL), ECN_Level

Warranty_Claims (13 rows)
  SerialNo, FailureDate, Symptom, MileageOrHours, Region, Severity

Supplier_Scorecard (5 rows)
  Supplier, Quality_Score, Tier, Incoming_FailRate, Warranty_ClaimRate,
  OnTimeDelivery_%, AvgLeadTime_Days, Process_Drift_Index, Premium_Service_Fit

Vendor_Engineering_Profile (5 rows)
  Supplier, COO, Engineering_Maturity, Process_Cpk,
  Design_Ownership, Engineering_Maturity_Score
  NOTE: Engineering_Maturity_Score ≠ Quality_Score — different sheets, different metrics.

COO_vs_Supplier (5 rows)
  Supplier, COO, Beats_COO_Avg (Yes/No), Incoming_FailRate, COO_Incoming_FailRate

Action_Playbook (5 rows)
  InsightType, TypicalAction, WhereItFits, SAP_or_MES_Touchpoint

JOIN MAP
--------
Warranty_Claims      → AsBuilt_Serial       on SerialNo  → gets supplier / COO / line
Constituent_BOM      → AsBuilt_Serial       on SerialNo  → gets line / shift / vendor
Constituent_BOM      → Incoming_QM          on (Component, Supplier, LotNo) → inspection results
Process_Measurements → AsBuilt_Serial       on SerialNo  → gets vendor / COO

Example multi-sheet join:
    abs_df = pd.read_excel(EXCEL_CACHE, sheet_name="AsBuilt_Serial")
    wc     = pd.read_excel(EXCEL_CACHE, sheet_name="Warranty_Claims")
    merged = wc.merge(abs_df[["SerialNo","VendorOfCriticalAssy"]], on="SerialNo", how="left")
    print(merged.groupby("VendorOfCriticalAssy").size())

KNOWN FACTS  (verified against data — never contradict these)
-------------------------------------------------------------
Total finished serials : 300
Total warranty claims  : 13
Total BOM rows         : 1200  (4 components × 300 serials)

Component → Suppliers (exclusive):
  SEAL-KIT     → SUP-A, SUP-C
  SENSOR-HALL  → SUP-C, SUP-E
  HOUSING      → SUP-D, SUP-E
  BEARING-SET  → SUP-B, SUP-D

Supplier tiers:
  Only Preferred-tier supplier: SUP-B
  SUP-A  Premium_Service_Fit = No

COO benchmark:
  Beats_COO_Avg = Yes: SUP-A, SUP-D only

HOW TO ANSWER
-------------
1. Call run_python to compute exactly what you need.
2. Lead with the direct answer in 1-2 sentences.
3. If a table helps, include a short one (max 6 rows).
4. Cite sheet + column inline, not as a separate section.
5. Stop as soon as the question is answered.
6. No bullet point recommendations unless explicitly asked.
7. No latent risk analysis unless explicitly asked.
8. No action items unless explicitly asked.
9. Maximum response length: 150 words.

STRICT RULES
------------
- Never state a number without computing it in this call.
- Never generate SAP codes that don't appear verbatim in
  Action_Playbook.SAP_or_MES_Touchpoint.
- Never say data is "missing" without querying all relevant sheets first.
- Never confuse Engineering_Maturity_Score (Vendor_Engineering_Profile)
  with Quality_Score (Supplier_Scorecard).
- Always join on SerialNo for cross-sheet questions.
- When joining Constituent_BOM to Incoming_QM, always deduplicate on
  (Component, Supplier, LotNo) BEFORE joining to avoid row multiplication:
      lots = bom_filtered[['Component','Supplier','LotNo']].drop_duplicates()
      iqm_matched = lots.merge(iqm, on=['Component','Supplier','LotNo'])
- When the question mentions "supplied units" or "supplier's units", default
  to VendorOfCriticalAssy in AsBuilt_Serial — do NOT query Constituent_BOM
  or Incoming_QM unless the question explicitly mentions BOM or components.
- After computing any percentage or count, sanity-check the result before
  answering: serial counts must not exceed 300, claim counts must not exceed
  13, percentages must be between 0–100%. If a number falls outside these
  bounds, recompute from scratch before stating the answer.
- CLAIM RATE CALCULATION — MANDATORY METHOD: always use AsBuilt_Serial unit
  counts as the denominator, never Constituent_BOM serial counts:
      ab    = pd.read_excel(EXCEL_CACHE, sheet_name="AsBuilt_Serial")
      wc    = pd.read_excel(EXCEL_CACHE, sheet_name="Warranty_Claims")
      units  = ab.groupby("VendorOfCriticalAssy")["SerialNo"].count()
      claims = ab.merge(wc[["SerialNo"]], on="SerialNo") \
                 .groupby("VendorOfCriticalAssy")["SerialNo"].count()
      claim_rate = (claims / units).fillna(0)
- No section headers: no "Finding:", "Evidence:", "Recommended Actions:",
  "Confidence %", or "Summary:".
"""

# ── Tool definition ───────────────────────────────────────────────────────────

_TOOL_DEF: Dict[str, Any] = {
    "name": "run_python",
    "description": (
        "Execute Python/pandas code against the production Excel workbook. "
        "EXCEL_PATH (string) is already in scope — use it directly. "
        "Every value you want to read must be printed. Returns captured stdout + errors."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to run. Import pandas inside the code. "
                    "Use print() for all output. "
                    "EXCEL_PATH is pre-defined — never hardcode a file path."
                ),
            }
        },
        "required": ["code"],
    },
}


# ── Code executor ─────────────────────────────────────────────────────────────

def _execute_python(code: str, excel_path: Path) -> str:
    """Run *code* in an isolated namespace; capture stdout + exceptions."""

    def _run() -> str:
        ns: Dict[str, Any] = {
                "EXCEL_PATH": excel_path,
                "EXCEL_CACHE": EXCEL_CACHE,
                "pd": pd,
                "__builtins__": __builtins__,
            }
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                exec(compile(code, "<run_python>", "exec"), ns)  # noqa: S102
            stdout = out_buf.getvalue().strip()
            stderr = err_buf.getvalue().strip()
            if stderr:
                return f"{stdout}\n[stderr]\n{stderr}".strip()
            return stdout or "(no output)"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=EXEC_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            return f"ERROR: execution timed out after {EXEC_TIMEOUT_S}s"


# ── Agent ─────────────────────────────────────────────────────────────────────

class ExcelQualityAgent:
    """
    Code-execution quality agent.
    Claude writes pandas code, we run it, Claude reads the output and answers.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        excel_path: Path = EXCEL_PATH,
    ) -> None:
        self.client = client
        self._excel_path = excel_path
        self._history: List[Dict[str, Any]] = []
        self._system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        # Keepalive disabled — burns ~900k tokens/day on hosted deployments
        # where the app idles 24/7. Cache misses add ~1s latency, which is
        # acceptable compared to continuous background spend.
        self._keepalive_active = False

    # -- Public ----------------------------------------------------------------

    def ask(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Non-streaming path (fallback). Uses stream() internally so the
        underlying HTTP connection is the same code path as ask_stream().
        Returns a dict compatible with agent_core routing.
        """
        messages: List[Dict[str, Any]] = [
            {"role": h["role"], "content": h["content"]}
            for h in self._history[-8:]
        ]
        messages.append({"role": "user", "content": question})

        tools_called: List[str] = []
        total_tokens = 0
        cache_hit = False
        final_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            with self.client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=self._system,
                tools=[_TOOL_DEF],
                messages=messages,
            ) as stream:
                resp = stream.get_final_message()

            usage = resp.usage
            print(f"cache usage: {usage}")
            logger.info("cache usage: %s", usage)
            total_tokens += int(usage.input_tokens + usage.output_tokens)
            if int(getattr(usage, "cache_read_input_tokens", 0)) > 0:
                cache_hit = True

            text_blocks = [b.text for b in resp.content if hasattr(b, "text")]

            if resp.stop_reason == "end_turn":
                final_text = "\n".join(text_blocks).strip()
                break

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        tools_called.append(block.name)
                        output = _execute_python(block.input["code"], self._excel_path)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = "\n".join(text_blocks).strip() or f"Stopped: {resp.stop_reason}"
                break
        else:
            final_text = "Maximum code-execution rounds reached without a final answer."

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": final_text})
        if len(self._history) > 12:
            self._history = self._history[-12:]

        return {
            "text":        final_text,
            "tools":       tools_called or ["run_python"],
            "confidence":  95,
            "suggestions": self._follow_ups(question),
            "cache_hit":   cache_hit,
            "tokens":      total_tokens,
            "_branch":     "excel_agent",
        }

    def ask_stream(
        self,
        question: str,
        session_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Generator for st.write_stream().
        Tool calls run synchronously; final text response streams token by token.
        *meta* is populated after exhaustion with tokens/cache_hit/ttft_ms/etc.
        """
        t0 = time.perf_counter()
        t_first_token: Optional[float] = None

        messages: List[Dict[str, Any]] = [
            {"role": h["role"], "content": h["content"]}
            for h in self._history[-8:]
        ]
        messages.append({"role": "user", "content": question})

        tools_called: List[str] = []
        total_tokens = 0
        cache_hit = False
        final_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            with self.client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=self._system,
                tools=[_TOOL_DEF],
                messages=messages,
            ) as stream:
                round_text = ""
                for chunk in stream.text_stream:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    round_text += chunk
                    yield chunk
                resp = stream.get_final_message()

            usage = resp.usage
            print(f"cache usage: {usage}")
            logger.info("cache usage: %s", usage)
            total_tokens += int(usage.input_tokens + usage.output_tokens)
            if int(getattr(usage, "cache_read_input_tokens", 0)) > 0:
                cache_hit = True

            final_text += round_text

            if resp.stop_reason == "end_turn":
                break

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        tools_called.append(block.name)
                        output = _execute_python(block.input["code"], self._excel_path)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        else:
            extra = "Maximum code-execution rounds reached."
            final_text += extra
            yield extra

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": final_text})
        if len(self._history) > 12:
            self._history = self._history[-12:]

        ttft_ms = round((t_first_token - t0) * 1000) if t_first_token else None
        logger.info("ttft=%sms  tokens=%d  cache_hit=%s", ttft_ms, total_tokens, cache_hit)

        if meta is not None:
            meta.update({
                "text":        final_text,
                "tools":       tools_called or ["run_python"],
                "confidence":  95,
                "suggestions": self._follow_ups(question),
                "cache_hit":   cache_hit,
                "tokens":      total_tokens,
                "ttft_ms":     ttft_ms,
                "_branch":     "excel_agent",
            })

    def clear(self) -> None:
        self._history.clear()

    def stop(self) -> None:
        self._keepalive_active = False

    # -- Internals -------------------------------------------------------------

    def _keepalive_loop(self) -> None:
        """Ping the API every 4 min with a 1-token request to keep the
        ephemeral cache warm (TTL = 5 min; ping at 4 min = safe margin)."""
        while self._keepalive_active:
            time.sleep(240)
            if not self._keepalive_active:
                break
            try:
                self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1,
                    system=self._system,
                    messages=[{"role": "user", "content": "."}],
                )
                logger.debug("Cache keepalive ping sent")
            except Exception as exc:
                logger.debug("Cache keepalive ping failed: %s", exc)

    @staticmethod
    def _follow_ups(question: str) -> List[str]:
        q = question.lower()
        if any(c in q for c in ["seal-kit", "sensor-hall", "bearing", "housing"]):
            comp = next(
                (c for c in ["SEAL-KIT", "SENSOR-HALL", "BEARING-SET", "HOUSING"]
                 if c.lower() in q),
                "this component",
            )
            return [
                f"Which lots of {comp} are currently HIGH risk?",
                f"What defect codes appear most in {comp}?",
                f"Which supplier should I prefer for {comp} in safety-critical builds?",
            ]
        if "l-778" in q or ("lot" in q and "risk" in q):
            return [
                "Which serials are affected by lot L-778?",
                "What corrective actions should I take for lot L-778?",
                "Is the supplier causing issues on other lots too?",
            ]
        if "sup-" in q or "supplier" in q:
            return [
                "Which supplier is best for safety-critical builds?",
                "Compare all suppliers by quality score",
                "Which suppliers beat their COO average?",
            ]
        if any(k in q for k in ["line", "drift", "process", "shift", "torque"]):
            return [
                "Which serials were built on LINE-2 Night shift?",
                "What is the leak fail rate on LINE-1 Day?",
                "Which line has the best overall quality record?",
            ]
        return [
            "What is the risk level of lot L-778?",
            "Which supplier shows highest precision for SEAL-KIT?",
            "Are there any process drift issues on our lines?",
        ]

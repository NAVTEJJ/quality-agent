"""
Follow-up suggestion generator (Phase 3, Step 4).

After every agent response we want to hand the user the next three
obvious questions -- entity-specific, not generic. The UI renders these
as clickable chips; the copilot should feel like a conversation that
threads naturally into the next investigation, not a Q&A kiosk.

Rules
-----
* Every follow-up cites real entities pulled from the current intent
  or from the tool_results. Generic prompts like "tell me more" are
  banned.
* Always returns exactly three suggestions (padded from a safe default
  list when the intent is too thin to generate that many).
* Never repeats the question that was just asked.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


_DEFAULT_FOLLOW_UPS = [
    "Tell me more about lot L-778",
    "Show me the LINE-2 Night shift issue in detail",
    "Which suppliers should I be watching?",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FollowUpGenerator:
    """Generates three entity-specific follow-up questions per turn."""

    def generate_follow_ups(
        self,
        intent: Any,
        entities: Optional[Dict[str, List[str]]] = None,
        tool_results: Optional[Sequence[Any]] = None,
    ) -> List[str]:
        entities = entities or {}
        tool_results = list(tool_results or [])
        intent_name = getattr(intent, "intent", None) or (
            intent.get("intent") if isinstance(intent, dict) else None
        ) or "GENERAL_INSIGHT"

        handler = {
            "LOT_RISK_QUERY":       self._lot_risk,
            "SUPPLIER_PROFILE":     self._supplier_profile,
            "PROCESS_DRIFT":        self._process_drift,
            "COO_ANALYSIS":         self._coo_analysis,
            "DRILL_DOWN":           self._drill_down,
            "INSPECTION_STRATEGY":  self._inspection_strategy,
            "SUPPLIER_COMPARE":     self._supplier_compare,
            "WARRANTY_TRACE":       self._warranty_trace,
            "GENERAL_INSIGHT":      self._general,
            "ACTION_REQUEST":       self._action_request,
        }.get(intent_name, self._general)

        suggestions = handler(entities, tool_results)
        return self._finalize(suggestions)

    # ==================================================================
    # Per-intent handlers
    # ==================================================================

    def _lot_risk(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        lot_no   = self._first(entities.get("lot_no")) or self._lot_from_tools(tool_results)
        supplier = self._supplier_from_tools(tool_results) or self._first(entities.get("supplier"))

        if lot_no and supplier:
            return [
                f"Show me all serials that used lot {lot_no}",
                f"What actions should I take immediately for lot {lot_no}?",
                f"Is {supplier} causing issues on other lots too?",
            ]
        if lot_no:
            return [
                f"Show me all serials that used lot {lot_no}",
                f"What actions should I take immediately for lot {lot_no}?",
                f"Who is the supplier for lot {lot_no} and what's their tier?",
            ]
        # No lot resolved -- fall back to the plant-wide risk view.
        return [
            "Which lots are currently HIGH risk?",
            "What's the riskiest lot in production right now?",
            "Show me the top 5 risky lots by composite score",
        ]

    def _process_drift(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        line  = self._first(entities.get("line"))  or self._line_from_tools(tool_results)
        shift = self._first(entities.get("shift")) or self._shift_from_tools(tool_results)

        if line and shift:
            return [
                f"Which serial numbers were built on {line} {shift} shift?",
                f"Has {line} {shift} drift been getting worse over time?",
                f"What's the torque calibration status for {line}?",
            ]
        if line:
            return [
                f"Which serial numbers were built on {line}?",
                f"Break down {line} performance by shift",
                f"What's the torque calibration status for {line}?",
            ]
        return [
            "Which line+shift combos have the highest torque fail rate?",
            "Has any drift signal been getting worse over time?",
            "Show me the serials on the drifting line/shift",
        ]

    def _supplier_profile(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        supplier = self._first(entities.get("supplier")) or self._supplier_from_tools(tool_results)
        top_sup = self._top_ranked_supplier(tool_results)
        if supplier and top_sup and supplier != top_sup:
            return [
                f"Compare {supplier} with {top_sup}",
                f"Which lots from {supplier} are currently in production?",
                f"Is {supplier} suitable for safety-critical builds?",
            ]
        if supplier:
            return [
                f"Compare {supplier} with the top-ranked supplier",
                f"Which lots from {supplier} are currently in production?",
                f"Is {supplier} suitable for safety-critical builds?",
            ]
        return [
            "Which suppliers are on the Watchlist tier right now?",
            "Who ranks #1 by composite quality score?",
            "Which suppliers are premium-service-fit?",
        ]

    def _coo_analysis(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        coo = self._first(entities.get("coo"))
        if coo:
            return [
                f"Which suppliers from {coo} beat their country average?",
                f"Is {coo}'s trend country-level or driven by specific suppliers?",
                f"Compare {coo} with the top-performing country",
            ]
        return [
            "Which country has the lowest incoming fail rate?",
            "Show me suppliers that beat their country average",
            "Is China's trend driven by a specific supplier?",
        ]

    def _drill_down(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        lot_no = self._first(entities.get("lot_no")) or self._lot_from_tools(tool_results)
        serial = self._first(entities.get("serial"))
        if serial:
            return [
                f"Show the warranty history for serial {serial}",
                f"Which lots fed the BOM of {serial}?",
                f"What process measurements did {serial} pass / fail?",
            ]
        if lot_no:
            return [
                f"Show the warranty claims linked to lot {lot_no}",
                f"Which line / shift built the serials from lot {lot_no}?",
                f"What actions should I take for lot {lot_no}?",
            ]
        return self._general(entities, tool_results)

    def _inspection_strategy(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        top_lot = self._lot_from_tools(tool_results)
        preferred = self._preferred_supplier_from_tools(tool_results)
        if top_lot and preferred:
            return [
                f"Why is lot {top_lot} at the top of the sampling list?",
                f"Can {preferred} move to audit-only inspection?",
                "Which suppliers should be escalated to Watchlist this week?",
            ]
        return [
            "Which HIGH-risk lots need 100% sampling right now?",
            "Which suppliers are eligible for reduced inspection?",
            "Which supplier/component combos should escalate to Watchlist?",
        ]

    def _supplier_compare(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        suppliers = list(entities.get("supplier") or [])
        if len(suppliers) >= 2:
            a, b = suppliers[0], suppliers[1]
            return [
                f"Decompose {a} vs {b} against their country averages",
                f"Which lots from {a} and {b} are in current production?",
                f"Is either {a} or {b} on the Watchlist tier?",
            ]
        a = suppliers[0] if suppliers else None
        if a:
            return [
                f"Compare {a} with the top-ranked supplier",
                f"Which lots from {a} are currently in production?",
                f"Is {a} suitable for safety-critical builds?",
            ]
        return [
            "Compare SUP-A and SUP-B for safety-critical builds",
            "Rank all suppliers by composite quality score",
            "Which suppliers beat their COO average?",
        ]

    def _warranty_trace(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        serial = self._first(entities.get("serial"))
        lot_no = self._first(entities.get("lot_no")) or self._lot_from_tools(tool_results)
        supplier = self._supplier_from_tools(tool_results)

        if serial and lot_no:
            return [
                f"Show every inspection run on lot {lot_no}",
                f"Who was the supplier for the failed component on {serial}?",
                f"Are there other warranty claims linked to lot {lot_no}?",
            ]
        if serial:
            return [
                f"Which lots fed {serial}'s BOM?",
                f"Show the process measurements for {serial}",
                f"Are there other field failures with the same symptom?",
            ]
        if supplier:
            return [
                f"Show every warranty claim tied back to {supplier}",
                f"Which of {supplier}'s lots have had inspection failures?",
                f"Compare {supplier}'s warranty rate against its COO average",
            ]
        return [
            "Show me the warranty claim spike on LOT-SENS-207",
            "Which suppliers have the highest warranty claim rate?",
            "Trace the most recent warranty claim to its root lot",
        ]

    def _general(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        return [
            "Tell me more about lot L-778",
            "Show me the LINE-2 Night shift issue in detail",
            "Which suppliers should I be watching?",
        ]

    def _action_request(
        self,
        entities: Dict[str, List[str]],
        tool_results: List[Any],
    ) -> List[str]:
        lot_no = self._first(entities.get("lot_no")) or self._lot_from_tools(tool_results)
        supplier = self._supplier_from_tools(tool_results) or self._first(entities.get("supplier"))
        if lot_no:
            return [
                f"Show me the containment checklist for lot {lot_no}",
                f"Which SAP transactions unblock lot {lot_no}?",
                (
                    f"Escalate {supplier} after lot {lot_no}?"
                    if supplier else
                    f"Who should own the 8D for lot {lot_no}?"
                ),
            ]
        return self._general(entities, tool_results)

    # ==================================================================
    # Tool-result scrapers
    # ==================================================================

    @staticmethod
    def _first(xs: Optional[List[str]]) -> Optional[str]:
        return xs[0] if xs else None

    @staticmethod
    def _result_dicts(tool_results: List[Any]) -> List[Dict[str, Any]]:
        dicts: List[Dict[str, Any]] = []
        for tr in tool_results:
            data = getattr(tr, "result_data", None)
            if isinstance(data, dict):
                dicts.append(data)
        return dicts

    def _lot_from_tools(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            if d.get("lot_no"):
                return str(d["lot_no"])
            li = d.get("lot_info")
            if isinstance(li, dict) and li.get("lot_no"):
                return str(li["lot_no"])
            # get_inspection_strategy output
            if isinstance(d.get("increase_sampling"), list) and d["increase_sampling"]:
                top = d["increase_sampling"][0]
                if isinstance(top, dict) and top.get("lot_no"):
                    return str(top["lot_no"])
        return None

    def _supplier_from_tools(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            if d.get("supplier"):
                return str(d["supplier"])
            for key in ("supplier_scorecard", "lot_info"):
                nested = d.get(key)
                if isinstance(nested, dict) and nested.get("supplier"):
                    return str(nested["supplier"])
        return None

    def _line_from_tools(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            signals = d.get("drift_signals") or d.get("records")
            if isinstance(signals, list) and signals:
                top = signals[0]
                if isinstance(top, dict) and top.get("line"):
                    return str(top["line"])
        return None

    def _shift_from_tools(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            signals = d.get("drift_signals") or d.get("records")
            if isinstance(signals, list) and signals:
                top = signals[0]
                if isinstance(top, dict) and top.get("shift"):
                    return str(top["shift"])
        return None

    def _top_ranked_supplier(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            rank = d.get("composite_rank")
            if rank == 1 and d.get("supplier"):
                return str(d["supplier"])
            rec = d.get("recommendation")
            if isinstance(rec, dict) and rec.get("winner"):
                return str(rec["winner"])
        return None

    def _preferred_supplier_from_tools(self, tool_results: List[Any]) -> Optional[str]:
        for d in self._result_dicts(tool_results):
            red = d.get("reduce_inspection")
            if isinstance(red, list) and red:
                top = red[0]
                if isinstance(top, dict) and top.get("supplier"):
                    return str(top["supplier"])
        return None

    # ==================================================================
    # Post-processing
    # ==================================================================

    @staticmethod
    def _finalize(xs: List[str]) -> List[str]:
        """Trim/pad to exactly three deduplicated, stripped suggestions."""
        seen: set[str] = set()
        cleaned: List[str] = []
        for s in xs:
            if not s:
                continue
            s2 = s.strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                cleaned.append(s2)
        for fallback in _DEFAULT_FOLLOW_UPS:
            if len(cleaned) >= 3:
                break
            if fallback not in seen:
                seen.add(fallback)
                cleaned.append(fallback)
        return cleaned[:3]


__all__ = ["FollowUpGenerator"]

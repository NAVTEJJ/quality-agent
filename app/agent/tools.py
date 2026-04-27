"""
Claude tool definitions for the Quality Agent (Phase 3, Step 2).

These are the exact ``tools=`` schemas passed to ``client.messages.create``.
Every tool has a tight, opinionated description -- Claude routes on these
strings, so generic wording produces wrong tool calls.

The shape of each dict is::

    {
        "name":         "...",
        "description":  "... when / why to call it ...",
        "input_schema": {JSON Schema object},
    }

Keep the catalogue in sync with :mod:`app.agent.tool_executor`:

* every tool name here **must** have an ``execute_*`` method on
  :class:`ToolExecutor`,
* every required input field must be handled explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Individual tools
# ---------------------------------------------------------------------------

GET_LOT_RISK: Dict[str, Any] = {
    "name": "get_lot_risk",
    "description": (
        "Retrieve risk score, fail rate, defect evidence, and warranty "
        "linkage for a specific lot number. Use when the user asks about "
        "lot quality, lot risk, lot-level defects, or whether a lot should "
        "be blocked / held. Returns composite risk score (0.0-1.0), risk "
        "tier (HIGH / MEDIUM / LOW), total inspections, total fails, "
        "inspection records, linked warranty claim count, and defect codes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lot_no": {
                "type": "string",
                "description": (
                    "The exact lot number as it appears in the warehouse, "
                    "e.g. 'L-778' or 'LOT-SEAL-215'. Case-sensitive match."
                ),
            },
            "include_warranty": {
                "type": "boolean",
                "description": (
                    "Whether to include the field-warranty linkage for "
                    "every serial consumed from this lot. Defaults to true; "
                    "set false for a lighter-weight response."
                ),
            },
        },
        "required": ["lot_no"],
    },
}


GET_SUPPLIER_PROFILE: Dict[str, Any] = {
    "name": "get_supplier_profile",
    "description": (
        "Retrieve a complete supplier scorecard including quality score, "
        "tier, incoming fail rate, warranty claim rate, process drift "
        "index, Cpk, engineering maturity, on-time delivery, COO context, "
        "and premium service fit. Use when the user asks about supplier "
        "performance, reliability, or suitability for a program. Does NOT "
        "compare suppliers -- use compare_suppliers for that."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_id": {
                "type": "string",
                "description": (
                    "The supplier identifier. Accepts the canonical name "
                    "('SUP-A', 'SUP-C', ...) or the integer id as a string."
                ),
            },
            "include_coo_decomposition": {
                "type": "boolean",
                "description": (
                    "Include this supplier's performance versus its COO "
                    "average (the beats_coo_avg signal). Defaults to true."
                ),
            },
        },
        "required": ["supplier_id"],
    },
}


GET_PROCESS_DRIFT: Dict[str, Any] = {
    "name": "get_process_drift",
    "description": (
        "Detect process drift signals by production line and shift. Returns "
        "torque and leak fail rates per line/shift combination with the "
        "is_drift_signal flag set when the torque_fail_rate exceeds the "
        "configured threshold. Use when the user asks about production "
        "line quality, shift performance, torque / leak issues, or process "
        "control."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "line": {
                "type": "string",
                "description": (
                    "Optional line filter, e.g. 'LINE-2'. Omit for all lines."
                ),
            },
            "shift": {
                "type": "string",
                "description": (
                    "Optional shift filter, e.g. 'Night' or 'Day'. Omit for "
                    "all shifts."
                ),
            },
            "only_flagged": {
                "type": "boolean",
                "description": (
                    "If true, return only line/shift combinations that "
                    "exceed the drift threshold. Defaults to false."
                ),
            },
        },
        "required": [],
    },
}


GET_COO_TREND: Dict[str, Any] = {
    "name": "get_coo_trend",
    "description": (
        "Retrieve country-of-origin quality trends including incoming fail "
        "rates, warranty claim rates, rank across countries, and the "
        "suppliers that beat / underperform the country average. ALWAYS "
        "call alongside get_supplier_profile when the user asks about a "
        "supplier from a given country, so you can decompose the signal "
        "and avoid COO stereotyping."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "coo": {
                "type": "string",
                "description": (
                    "Optional country filter, e.g. 'China'. Case-insensitive. "
                    "Omit to return the trend for every country."
                ),
            },
        },
        "required": [],
    },
}


GET_DRILL_DOWN: Dict[str, Any] = {
    "name": "get_drill_down",
    "description": (
        "Retrieve the complete traceability chain for a lot or serial: "
        "lot -> inspection records -> affected serials -> process "
        "measurements -> warranty outcomes -> supplier scorecard -> COO "
        "context. Use for deep investigation, 'show me the evidence', "
        "root-cause walkthroughs, or when the user has already been shown "
        "a headline and is asking for the underlying data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lot_no": {
                "type": "string",
                "description": "Lot number, e.g. 'L-778'. Optional if serial_no is given.",
            },
            "serial_no": {
                "type": "string",
                "description": (
                    "Serial number, e.g. 'SR20260008'. Optional if lot_no "
                    "is given. When both are present, lot_no wins."
                ),
            },
            "depth": {
                "type": "string",
                "enum": ["full", "lot_only", "serial_only"],
                "description": (
                    "'full' = entire lot->serial->warranty chain (default "
                    "for lot queries); 'lot_only' = lot header + "
                    "inspections + serial list (no per-serial expansion); "
                    "'serial_only' = serial header + process + warranty + "
                    "BOM back-references."
                ),
            },
        },
        "required": [],
    },
}


GET_INSPECTION_STRATEGY: Dict[str, Any] = {
    "name": "get_inspection_strategy",
    "description": (
        "Get prioritised inspection recommendations for the plant: which "
        "lots need increased sampling, which suppliers are eligible for "
        "reduced / audit-only inspection, and which supplier-component "
        "combos should be escalated to Watchlist. Use when the user asks "
        "where to focus QC effort, how to reallocate inspection capacity, "
        "or which lots to prioritise."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_threshold": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "ALL"],
                "description": (
                    "Filter the 'increase_sampling' list by tier. 'HIGH' = "
                    "HIGH-risk lots only (default); 'MEDIUM' = HIGH + "
                    "MEDIUM; 'ALL' = also include LOW."
                ),
            },
        },
        "required": [],
    },
}


GET_ACTION_PLAYBOOK: Dict[str, Any] = {
    "name": "get_action_playbook",
    "description": (
        "Get recommended actions with SAP / MES touchpoints for a given "
        "insight type. Returns the step-by-step actions and the exact "
        "SAP transaction codes (QA32, QE51N, QM01, etc.) that the quality "
        "team uses. ALWAYS call this when you are about to make a "
        "recommendation -- do NOT invent actions from training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "insight_type": {
                "type": "string",
                "description": (
                    "The type of insight you need actions for. Examples: "
                    "'High lot defect rate', 'Warranty claim spike', "
                    "'Process drift', 'Preferred supplier', 'Premium "
                    "engineering supplier fit'. Matches playbook rows by "
                    "substring."
                ),
            },
            "supplier_tier": {
                "type": "string",
                "description": (
                    "Optional tier filter: 'Watchlist' | 'Standard' | "
                    "'Preferred'. When given, tier-specific actions are "
                    "appended to the base playbook actions."
                ),
            },
        },
        "required": ["insight_type"],
    },
}


SEARCH_INSIGHTS: Dict[str, Any] = {
    "name": "search_insights",
    "description": (
        "Search the pre-computed AI insight catalogue for pattern matches. "
        "Returns relevant insights from ref_ai_insights with pattern, "
        "evidence, guidance, and suggested actionables. Use as the FIRST "
        "call for broad or open-ended questions about current risks / "
        "opportunities (e.g. 'what are the biggest risks right now')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Free-text search query. Matched case-insensitively "
                    "against pattern_detected, evidence, and ai_guidance."
                ),
            },
            "risk_type": {
                "type": "string",
                "enum": ["Risk", "Opportunity", "Both"],
                "description": (
                    "Filter by the risk_or_opportunity column. Defaults "
                    "to 'Both' when omitted."
                ),
            },
        },
        "required": ["query"],
    },
}


COMPARE_SUPPLIERS: Dict[str, Any] = {
    "name": "compare_suppliers",
    "description": (
        "Produce a side-by-side comparison of two or more suppliers across "
        "every quality dimension: fail rate, warranty rate, engineering "
        "maturity, Cpk, COO context, tier, and composite score. Use ONLY "
        "when the user explicitly compares suppliers or asks which of a "
        "named set is best for a given use case. For single-supplier "
        "queries, prefer get_supplier_profile."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_ids": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string"},
                "description": (
                    "List of supplier identifiers (canonical names like "
                    "'SUP-A'). Must contain at least two."
                ),
            },
            "use_case": {
                "type": "string",
                "description": (
                    "Optional target use-case, e.g. 'safety-critical', "
                    "'cost-optimised', 'new platform'. Shapes the winner "
                    "recommendation: safety-critical weights Cpk + "
                    "engineering maturity higher."
                ),
            },
        },
        "required": ["supplier_ids"],
    },
}


GET_MATERIAL_VENDORS: Dict[str, Any] = {
    "name": "get_material_vendors",
    "description": (
        "Look up WHICH vendors actually supply a specific material or component, "
        "then return their precision scores, fail rates, Cpk, and quality scores "
        "side-by-side. ALWAYS call this FIRST when the user asks about vendors "
        "for a specific material (e.g. 'for material SEAL-KIT which vendor shows "
        "high precision'). Do NOT guess which suppliers make a component — use "
        "this tool to get the authoritative list from the warehouse."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "material_name": {
                "type": "string",
                "description": (
                    "The component or material name to look up, e.g. 'SEAL-KIT', "
                    "'SENSOR-HALL', 'BEARING'. Matched case-insensitively with "
                    "LIKE search against dim_component.component_name."
                ),
            },
            "metric": {
                "type": "string",
                "enum": ["precision", "quality", "fail_rate", "cpk", "warranty"],
                "description": (
                    "The primary metric to rank vendors by. Defaults to 'precision'. "
                    "precision = precision_score (consistency/repeatability); "
                    "quality = quality_score (0-100 composite); "
                    "fail_rate = incoming_fail_rate (lower is better); "
                    "cpk = process_cpk (capability index); "
                    "warranty = warranty_claim_rate (lower is better)."
                ),
            },
        },
        "required": ["material_name"],
    },
}


GET_WARRANTY_TRACE: Dict[str, Any] = {
    "name": "get_warranty_trace",
    "description": (
        "Trace a warranty claim backwards to its likely root cause: claim "
        "-> finished serial -> BOM -> consumed supplier lots -> supplier "
        "-> incoming-inspection history. Use for field failure analysis, "
        "'why did this unit fail?' questions, and warranty-to-supplier "
        "root-cause investigations. Accepts any of serial_no, claim_id, "
        "or supplier_id -- at least one is required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "serial_no": {
                "type": "string",
                "description": "Finished-goods serial number, e.g. 'SR20260008'.",
            },
            "claim_id": {
                "type": "string",
                "description": "Warranty claim identifier, e.g. 'WC910002'.",
            },
            "supplier_id": {
                "type": "string",
                "description": (
                    "Supplier id or canonical name, e.g. 'SUP-C'. When "
                    "given, returns every warranty claim linked back to "
                    "a lot supplied by this vendor."
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

CLAUDE_TOOLS: List[Dict[str, Any]] = [
    GET_LOT_RISK,
    GET_SUPPLIER_PROFILE,
    GET_PROCESS_DRIFT,
    GET_COO_TREND,
    GET_DRILL_DOWN,
    GET_INSPECTION_STRATEGY,
    GET_ACTION_PLAYBOOK,
    SEARCH_INSIGHTS,
    COMPARE_SUPPLIERS,
    GET_WARRANTY_TRACE,
    GET_MATERIAL_VENDORS,
]

TOOL_NAMES: List[str] = [t["name"] for t in CLAUDE_TOOLS]


__all__ = [
    "CLAUDE_TOOLS",
    "TOOL_NAMES",
    "GET_LOT_RISK",
    "GET_SUPPLIER_PROFILE",
    "GET_PROCESS_DRIFT",
    "GET_COO_TREND",
    "GET_DRILL_DOWN",
    "GET_INSPECTION_STRATEGY",
    "GET_ACTION_PLAYBOOK",
    "SEARCH_INSIGHTS",
    "COMPARE_SUPPLIERS",
    "GET_WARRANTY_TRACE",
    "GET_MATERIAL_VENDORS",
]

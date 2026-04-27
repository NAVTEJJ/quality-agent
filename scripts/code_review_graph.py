"""
Code Review Graph — AI Quality Inspection Copilot.

Generates a self-contained interactive HTML file showing:
  - Module dependency graph (who imports whom)
  - Per-file code metrics (LOC, functions, classes, complexity)
  - Layer coloring (agent / frontend / services / ingestion / tests / scripts)
  - Hover tooltips with full metrics
  - Sidebar summary with top files by size and hotspot detection

Usage:
    python scripts/code_review_graph.py
    python scripts/code_review_graph.py --out docs/code_graph.html

Output opens in any browser — no server or extra dependencies required.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]


# ── File collection ───────────────────────────────────────────────────────────

_SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", ".pytest_cache", "node_modules"}
_SKIP_NAMES = {"__init__.py"}


def _collect_files() -> List[Path]:
    files = []
    for p in sorted(_ROOT.rglob("*.py")):
        if any(d in p.parts for d in _SKIP_DIRS):
            continue
        if p.name in _SKIP_NAMES:
            continue
        files.append(p.relative_to(_ROOT))
    return files


# ── Code metrics ──────────────────────────────────────────────────────────────

def _metrics(path: Path) -> Dict[str, Any]:
    abs_path = _ROOT / path
    try:
        src = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"loc": 0, "functions": 0, "classes": 0, "imports": [], "complexity": 0}

    lines = src.splitlines()
    loc   = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))

    imports: List[str] = []
    functions = 0
    classes   = 0
    complexity = 1  # base complexity

    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
            elif isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
            # Cyclomatic complexity: count branching nodes
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                   ast.With, ast.Assert, ast.BoolOp)):
                complexity += 1
    except SyntaxError:
        pass

    return {
        "loc":        loc,
        "functions":  functions,
        "classes":    classes,
        "imports":    imports,
        "complexity": complexity,
    }


# ── Layer classification ──────────────────────────────────────────────────────

_LAYER_MAP = {
    "app/agent":             ("Agent",     "#58A6FF", "#1a2a3a"),
    "app/frontend":          ("Frontend",  "#BC8CFF", "#2a1a3a"),
    "app/services":          ("Services",  "#3FB950", "#1a2a1a"),
    "app/ingestion":         ("Ingestion", "#D29922", "#2a1e00"),
    "app/models":            ("Models",    "#F0883E", "#2a1800"),
    "app/core":              ("Core",      "#79C0FF", "#152030"),
    "configs":               ("Config",    "#8B949E", "#1a1e22"),
    "tests":                 ("Tests",     "#56D364", "#102010"),
    "scripts":               ("Scripts",   "#FFA657", "#201800"),
}

def _layer(path: Path) -> Tuple[str, str, str]:
    """Return (layer_name, border_color, bg_color)."""
    s = str(path).replace("\\", "/")
    for prefix, info in _LAYER_MAP.items():
        if s.startswith(prefix):
            return info
    return ("Other", "#484F58", "#1a1a1a")


# ── Dependency edge resolution ────────────────────────────────────────────────

def _module_to_path(module: str, all_paths: List[Path]) -> Optional[str]:
    """Map a dotted import string to a relative path string, or None if external."""
    parts = module.replace(".", "/")
    candidates = [
        f"{parts}.py",
        f"{parts}/__init__.py",
    ]
    for c in candidates:
        if any(str(p).replace("\\", "/") == c for p in all_paths):
            return c.replace("/__init__.py", "/__init__")
    # Match by suffix
    for p in all_paths:
        ps = str(p).replace("\\", "/")
        if ps == parts + ".py" or ps.startswith(parts + "/"):
            return ps
    return None


def _build_graph(files: List[Path]) -> Tuple[List[Dict], List[Dict]]:
    file_strs = [str(f).replace("\\", "/") for f in files]
    nodes: List[Dict] = []
    edges: List[Dict] = []
    edge_set: Set[Tuple[str, str]] = set()

    for path in files:
        key   = str(path).replace("\\", "/")
        m     = _metrics(path)
        layer, color, bg = _layer(path)

        # Node size = sqrt(LOC) so very large files don't dominate
        size = max(18, min(60, int(m["loc"] ** 0.55)))

        # Complexity color override: very complex files get red border
        border = color
        if m["complexity"] > 40:
            border = "#F85149"
        elif m["complexity"] > 20:
            border = "#D29922"

        nodes.append({
            "id":         key,
            "label":      path.name.replace(".py", ""),
            "title":      (
                f"<b>{key}</b><br>"
                f"Layer: {layer}<br>"
                f"LOC: {m['loc']}<br>"
                f"Functions: {m['functions']}<br>"
                f"Classes: {m['classes']}<br>"
                f"Complexity: {m['complexity']}<br>"
                f"Imports: {len(m['imports'])}"
            ),
            "color": {
                "background": bg,
                "border":     border,
                "highlight":  {"background": "#30363D", "border": color},
                "hover":      {"background": "#21262D", "border": color},
            },
            "font":  {"color": color, "size": 12},
            "size":  size,
            "layer": layer,
            "loc":   m["loc"],
            "functions": m["functions"],
            "classes":   m["classes"],
            "complexity": m["complexity"],
            "shape": "dot",
        })

        # Edges from imports
        for imp in m["imports"]:
            target = _module_to_path(imp, files)
            if target and target != key:
                # normalise target to file_strs
                target_norm = target.replace("/__init__", "/__init__.py")
                if target_norm in file_strs:
                    edge_key = (key, target_norm)
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append({
                            "from":   key,
                            "to":     target_norm,
                            "arrows": "to",
                            "color":  {"color": "#30363D", "highlight": color, "hover": color},
                            "width":  1,
                        })

    return nodes, edges


# ── Summary stats ─────────────────────────────────────────────────────────────

def _summary(nodes: List[Dict]) -> Dict[str, Any]:
    total_loc   = sum(n["loc"]        for n in nodes)
    total_fns   = sum(n["functions"]  for n in nodes)
    total_cls   = sum(n["classes"]    for n in nodes)

    by_layer: Dict[str, int] = {}
    for n in nodes:
        by_layer[n["layer"]] = by_layer.get(n["layer"], 0) + n["loc"]

    top_loc = sorted(nodes, key=lambda n: n["loc"], reverse=True)[:8]
    top_cpx = sorted(nodes, key=lambda n: n["complexity"], reverse=True)[:5]

    hotspots = [n for n in nodes if n["complexity"] > 30 or n["loc"] > 500]

    return {
        "total_loc":   total_loc,
        "total_fns":   total_fns,
        "total_cls":   total_cls,
        "total_files": len(nodes),
        "by_layer":    by_layer,
        "top_loc":     [{"id": n["id"], "loc": n["loc"]} for n in top_loc],
        "top_cpx":     [{"id": n["id"], "cpx": n["complexity"]} for n in top_cpx],
        "hotspots":    [{"id": n["id"], "loc": n["loc"], "cpx": n["complexity"]} for n in hotspots],
    }


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Code Review Graph — AI Quality Inspection Copilot</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0D1117; color: #C9D1D9; font-family: -apple-system, BlinkMacSystemFont,
          'Segoe UI', monospace; font-size: 13px; display: flex; flex-direction: column;
          height: 100vh; overflow: hidden; }}

  header {{ background: #161B22; border-bottom: 1px solid #30363D;
            padding: 0.6rem 1.2rem; display: flex; align-items: center; gap: 1rem; flex-shrink: 0; }}
  header h1 {{ font-size: 0.95rem; font-weight: 700; color: #58A6FF; }}
  header span {{ font-size: 0.75rem; color: #8B949E; }}

  .main {{ display: flex; flex: 1; overflow: hidden; }}

  #graph {{ flex: 1; background: #0D1117; }}

  aside {{ width: 280px; background: #161B22; border-left: 1px solid #30363D;
           overflow-y: auto; padding: 1rem; flex-shrink: 0; }}
  aside h2 {{ font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.08em; color: #484F58; margin-bottom: 0.75rem;
              padding-bottom: 0.4rem; border-bottom: 1px solid #30363D; }}

  .metric-row {{ display: flex; justify-content: space-between; padding: 3px 0;
                 font-size: 0.75rem; border-bottom: 1px solid #21262D; }}
  .metric-row .val {{ color: #58A6FF; font-weight: 600; }}

  .layer-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                margin-right: 5px; vertical-align: middle; }}

  .file-row {{ font-size: 0.72rem; padding: 4px 0; display: flex;
               justify-content: space-between; border-bottom: 1px solid #1a1e24; cursor: pointer; }}
  .file-row:hover {{ color: #58A6FF; }}
  .file-row .name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                      max-width: 160px; color: #8B949E; }}
  .file-row .badge {{ font-size: 0.68rem; border-radius: 3px; padding: 1px 5px;
                       background: #21262D; color: #8B949E; white-space: nowrap; }}

  .hotspot {{ color: #F85149; }}
  .warn     {{ color: #D29922; }}

  section {{ margin-bottom: 1.25rem; }}

  .legend-item {{ display: flex; align-items: center; font-size: 0.72rem;
                  color: #8B949E; padding: 2px 0; }}

  #node-detail {{ background: #0D1117; border: 1px solid #30363D; border-radius: 6px;
                  padding: 0.75rem; margin-top: 0.5rem; display: none; }}
  #node-detail h3 {{ font-size: 0.8rem; color: #58A6FF; margin-bottom: 0.5rem; }}
  #node-detail .nd-row {{ font-size: 0.72rem; padding: 2px 0; display: flex;
                           justify-content: space-between; }}
  #node-detail .nd-row .k {{ color: #484F58; }}
  #node-detail .nd-row .v {{ color: #C9D1D9; }}
</style>
</head>
<body>

<header>
  <h1>Code Review Graph — AI Quality Inspection Copilot</h1>
  <span>{total_files} files &nbsp;|&nbsp; {total_loc:,} LOC &nbsp;|&nbsp;
        {total_fns} functions &nbsp;|&nbsp; {total_cls} classes</span>
  <span style="margin-left:auto;font-size:0.7rem;color:#484F58;">
    Node size = LOC &nbsp;·&nbsp; Red border = high complexity &nbsp;·&nbsp; Click node for details
  </span>
</header>

<div class="main">
  <div id="graph"></div>

  <aside>
    <!-- Node detail (shown on click) -->
    <div id="node-detail">
      <h3 id="nd-title">—</h3>
      <div class="nd-row"><span class="k">Layer</span>     <span class="v" id="nd-layer">—</span></div>
      <div class="nd-row"><span class="k">LOC</span>       <span class="v" id="nd-loc">—</span></div>
      <div class="nd-row"><span class="k">Functions</span> <span class="v" id="nd-fns">—</span></div>
      <div class="nd-row"><span class="k">Classes</span>   <span class="v" id="nd-cls">—</span></div>
      <div class="nd-row"><span class="k">Complexity</span><span class="v" id="nd-cpx">—</span></div>
    </div>

    <section>
      <h2>Layer breakdown (LOC)</h2>
      {layer_rows}
    </section>

    <section>
      <h2>Largest files</h2>
      {top_loc_rows}
    </section>

    <section>
      <h2>⚠️ Hotspots (LOC &gt; 500 or complexity &gt; 30)</h2>
      {hotspot_rows}
    </section>

    <section>
      <h2>Highest complexity</h2>
      {top_cpx_rows}
    </section>

    <section>
      <h2>Legend</h2>
      {legend_rows}
    </section>
  </aside>
</div>

<script>
const NODES = {nodes_json};
const EDGES = {edges_json};
const NODE_MAP = {{}};
NODES.forEach(n => NODE_MAP[n.id] = n);

const container = document.getElementById('graph');
const data      = {{ nodes: new vis.DataSet(NODES), edges: new vis.DataSet(EDGES) }};
const options   = {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{ gravitationalConstant: -60, centralGravity: 0.01,
                          springLength: 140, springConstant: 0.06, damping: 0.5 }},
    stabilization: {{ iterations: 280 }},
  }},
  interaction: {{ hover: true, tooltips: true, navigationButtons: false,
                  zoomView: true, dragView: true }},
  nodes: {{ borderWidth: 2, borderWidthSelected: 3 }},
  edges: {{ smooth: {{ type: 'continuous' }}, selectionWidth: 2 }},
}};
const network = new vis.Network(container, data, options);

// Click → show detail panel
network.on('click', params => {{
  const detail = document.getElementById('node-detail');
  if (!params.nodes.length) {{ detail.style.display = 'none'; return; }}
  const n = NODE_MAP[params.nodes[0]];
  if (!n) return;
  document.getElementById('nd-title').textContent = n.id;
  document.getElementById('nd-layer').textContent = n.layer;
  document.getElementById('nd-loc').textContent   = n.loc.toLocaleString();
  document.getElementById('nd-fns').textContent   = n.functions;
  document.getElementById('nd-cls').textContent   = n.classes;
  const cpxEl = document.getElementById('nd-cpx');
  cpxEl.textContent = n.complexity;
  cpxEl.style.color = n.complexity > 40 ? '#F85149' : n.complexity > 20 ? '#D29922' : '#3FB950';
  detail.style.display = 'block';
}});

// After stabilization, disable physics for performance
network.on('stabilizationIterationsDone', () => {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});
</script>
</body>
</html>
"""


def _layer_rows(by_layer: Dict[str, int]) -> str:
    total = max(sum(by_layer.values()), 1)
    rows  = []
    for layer, loc in sorted(by_layer.items(), key=lambda x: -x[1]):
        pct  = loc / total * 100
        _, clr, _ = next(
            (v for k, v in _LAYER_MAP.items() if v[0] == layer),
            ("", "#8B949E", ""),
        )
        bar_w = max(1, int(pct * 1.2))
        rows.append(
            f'<div class="metric-row">'
            f'<span><span class="layer-dot" style="background:{clr}"></span>{layer}</span>'
            f'<span class="val">{loc:,} LOC</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _file_rows(items: List[Dict], value_key: str, value_label: str,
               hotspot: bool = False) -> str:
    rows = []
    for item in items:
        name   = item["id"].split("/")[-1].replace(".py", "")
        val    = item[value_key]
        cls    = "hotspot" if hotspot and (item.get("cpx", 0) > 40 or item.get("loc", 0) > 700) \
                 else "warn" if hotspot else ""
        rows.append(
            f'<div class="file-row {cls}">'
            f'<span class="name" title="{item["id"]}">{name}</span>'
            f'<span class="badge">{val} {value_label}</span>'
            f'</div>'
        )
    return "\n".join(rows) if rows else '<div style="color:#484F58;font-size:0.72rem;">None detected</div>'


def _legend_rows() -> str:
    rows = []
    for _, (layer, clr, _) in _LAYER_MAP.items():
        rows.append(
            f'<div class="legend-item">'
            f'<span class="layer-dot" style="background:{clr}"></span>'
            f'{layer}'
            f'</div>'
        )
    rows.append(
        '<div class="legend-item" style="margin-top:6px;">'
        '<span class="layer-dot" style="background:#F85149;border-radius:2px;"></span>'
        'High complexity border (>40)'
        '</div>'
    )
    rows.append(
        '<div class="legend-item">'
        '<span class="layer-dot" style="background:#D29922;border-radius:2px;"></span>'
        'Medium complexity border (>20)'
        '</div>'
    )
    return "\n".join(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive code review graph.")
    parser.add_argument("--out", default="docs/code_review_graph.html",
                        help="Output HTML path (relative to project root)")
    args = parser.parse_args()

    out_path = _ROOT / args.out

    print("Collecting files...")
    files = _collect_files()
    print(f"  {len(files)} Python files found")

    print("Building dependency graph...")
    nodes, edges = _build_graph(files)
    print(f"  {len(nodes)} nodes, {len(edges)} edges")

    summ = _summary(nodes)

    print("Rendering HTML...")
    html = _HTML_TEMPLATE.format(
        total_files = summ["total_files"],
        total_loc   = summ["total_loc"],
        total_fns   = summ["total_fns"],
        total_cls   = summ["total_cls"],
        nodes_json  = json.dumps(nodes,  ensure_ascii=False),
        edges_json  = json.dumps(edges,  ensure_ascii=False),
        layer_rows  = _layer_rows(summ["by_layer"]),
        top_loc_rows= _file_rows(summ["top_loc"], "loc", "LOC"),
        hotspot_rows= _file_rows(summ["hotspots"], "loc", "LOC", hotspot=True),
        top_cpx_rows= _file_rows(summ["top_cpx"], "cpx", "CC"),
        legend_rows = _legend_rows(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print()
    print(f"  Output : {out_path}")
    print(f"  Size   : {out_path.stat().st_size // 1024} KB")
    print()
    print("Summary")
    print(f"  Files      : {summ['total_files']}")
    print(f"  Total LOC  : {summ['total_loc']:,}")
    print(f"  Functions  : {summ['total_fns']}")
    print(f"  Classes    : {summ['total_cls']}")
    print(f"  Edges      : {len(edges)}")
    print()
    if summ["hotspots"]:
        print("  Hotspots (review priority):")
        for h in summ["hotspots"]:
            print(f"    {h['id']}  (LOC={h['loc']}, CC={h['cpx']})")
    else:
        print("  No hotspots detected.")
    print()
    print(f"Open in browser:  {out_path.name}")


if __name__ == "__main__":
    main()

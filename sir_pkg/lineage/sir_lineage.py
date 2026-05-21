#!/usr/bin/env python3
"""
sir_lineage.py  —  ArXivist SIR Lineage Graph
===============================================

Builds a citation and architectural inheritance graph across all SIRs in the
ArXivist registry. Produces an interactive HTML visualisation and a machine-
readable graph JSON.

TWO TYPES OF EDGES
------------------
  Inheritance edge  (solid)
      Two papers share high architectural similarity (diff score ≥ threshold).
      Weight = SIR similarity score. Interpreted as: "B likely builds on A."

  Citation edge  (dashed)
      Paper B's SIR references paper A by name in its title, abstract, key
      claims, or implementation assumptions. Extracted from text, not metadata.

GRAPH OUTPUTS
-------------
  --format html     Interactive force-directed graph in a self-contained HTML
                    file. Nodes are papers, edges are relationships. Hover for
                    SIR details. Click to highlight neighbourhood. (default)

  --format json     Machine-readable graph: nodes + edges with all attributes.

  --format dot      Graphviz DOT format for rendering with graphviz tools.

  --format report   Markdown report: most connected papers, strongest
                    inheritance chains, isolated papers (no edges).

USAGE
-----
  # Build lineage graph from the full registry
  python sir_lineage.py

  # Set similarity threshold for inheritance edges (default: 0.45)
  python sir_lineage.py --threshold 0.60

  # Filter to one domain
  python sir_lineage.py --domain Finance

  # Output formats
  python sir_lineage.py --format html --out lineage.html
  python sir_lineage.py --format json --out lineage.json
  python sir_lineage.py --format dot  --out lineage.dot
  python sir_lineage.py --format report

  # Ego graph — show only papers within N hops of a given paper
  python sir_lineage.py --ego arxiv_1706_03762 --hops 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Import the diff engine — must be in the same directory or on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "diff"))
try:
    from sir_diff import compute_diff, load_sir  # type: ignore
    DIFF_AVAILABLE = True
except ImportError:
    DIFF_AVAILABLE = False


# ============================================================
# 1. DATA STRUCTURES
# ============================================================

@dataclass
class Node:
    paper_id: str
    title: str
    domain: str
    overall_confidence: float
    n_modules: int
    primary_metric: Optional[str]
    has_comparison: bool = False


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str          # "inheritance" or "citation"
    weight: float           # similarity score (inheritance) or 1.0 (citation)
    label: str              # human-readable description


@dataclass
class LineageGraph:
    nodes: list[Node]
    edges: list[Edge]
    threshold: float
    n_pairs_evaluated: int


# ============================================================
# 2. REGISTRY LOADING
# ============================================================

def load_registry(registry_dir: str) -> list[dict]:
    registry_path = Path(registry_dir)
    sirs = []
    for sir_path in sorted(registry_path.rglob("sir.json")):
        if "versions" in sir_path.parts:
            continue
        try:
            with open(sir_path) as f:
                sir = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "paper_id" not in sir:
            sir["paper_id"] = sir_path.parent.name
        # Load comparison report if present
        comp_path = sir_path.parent.parent.parent / "paper-repos" / sir.get("paper_id", "") / "comparison" / "reproducibility_score.json"
        sir["_has_comparison"] = comp_path.exists()
        sirs.append(sir)
    return sirs


# ============================================================
# 3. NODE EXTRACTION
# ============================================================

def sir_to_node(sir: dict) -> Node:
    prov = sir.get("provenance", {})
    arch = sir.get("architecture", {})
    ep = sir.get("evaluation_protocol", {})
    conf = sir.get("confidence_annotations", {})

    primary_metric = next(
        (r.get("metric") for r in ep.get("reported_results", []) if r.get("is_primary")),
        None,
    )
    domain = prov.get("subject_domain") or prov.get("domain", "Unknown")

    return Node(
        paper_id=sir.get("paper_id", "?"),
        title=prov.get("title", sir.get("paper_id", "?")),
        domain=str(domain),
        overall_confidence=conf.get("overall_sir_confidence", 0.0),
        n_modules=len(arch.get("modules", [])),
        primary_metric=primary_metric,
        has_comparison=sir.get("_has_comparison", False),
    )


# ============================================================
# 4. EDGE EXTRACTION
# ============================================================

def _extract_citation_edges(sirs: list[dict]) -> list[Edge]:
    """Extract citation edges by detecting paper title mentions in SIR text."""
    # Build title → paper_id lookup
    title_map: dict[str, str] = {}
    for sir in sirs:
        title = sir.get("provenance", {}).get("title", "")
        if title:
            # Use first 5+ word subsequence as a fingerprint
            words = re.findall(r"[a-z0-9]+", title.lower())
            if len(words) >= 3:
                key = " ".join(words[:5])
                title_map[key] = sir.get("paper_id", "")

    edges = []
    for sir in sirs:
        pid_b = sir.get("paper_id", "")
        prov = sir.get("provenance", {})

        # Search text fields for mentions of other papers
        search_text = " ".join([
            prov.get("abstract", ""),
            " ".join(prov.get("key_claims", [])),
            " ".join(
                a.get("assumption", "") + " " + a.get("basis", "")
                for a in sir.get("implementation_assumptions", [])
            ),
        ]).lower()

        for title_key, pid_a in title_map.items():
            if pid_a == pid_b:
                continue
            if title_key in search_text:
                edges.append(Edge(
                    source=pid_a,
                    target=pid_b,
                    edge_type="citation",
                    weight=1.0,
                    label="cited in text",
                ))

    return edges


def _extract_inheritance_edges(
    sirs: list[dict], threshold: float
) -> tuple[list[Edge], int]:
    """Extract inheritance edges by running pairwise SIR diffs."""
    if not DIFF_AVAILABLE:
        print(
            "Warning: sir_diff.py not found. Inheritance edges will not be computed.\n"
            "Place sir_diff.py in ../diff/ relative to this script.",
            file=sys.stderr,
        )
        return [], 0

    edges = []
    n_pairs = 0
    n = len(sirs)

    for i in range(n):
        for j in range(i + 1, n):
            sir_a = sirs[i]
            sir_b = sirs[j]
            n_pairs += 1

            try:
                diff = compute_diff(sir_a, sir_b)
            except Exception:
                continue

            if diff.overall_similarity >= threshold:
                # Determine directionality heuristically:
                # The paper with the earlier arXiv ID is likely the source.
                pid_a = sir_a.get("paper_id", "")
                pid_b = sir_b.get("paper_id", "")
                source, target = (pid_a, pid_b) if pid_a <= pid_b else (pid_b, pid_a)

                edges.append(Edge(
                    source=source,
                    target=target,
                    edge_type="inheritance",
                    weight=round(diff.overall_similarity, 4),
                    label=f"similarity={diff.overall_similarity:.3f}",
                ))

    return edges, n_pairs


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    sirs: list[dict],
    threshold: float = 0.45,
    domain_filter: Optional[str] = None,
) -> LineageGraph:

    if domain_filter:
        sirs = [
            s for s in sirs
            if (
                s.get("provenance", {}).get("subject_domain", "") or
                s.get("provenance", {}).get("domain", "")
            ).lower() == domain_filter.lower()
        ]
        print(f"Domain filter '{domain_filter}': {len(sirs)} SIRs", file=sys.stderr)

    nodes = [sir_to_node(s) for s in sirs]
    citation_edges = _extract_citation_edges(sirs)
    inheritance_edges, n_pairs = _extract_inheritance_edges(sirs, threshold)

    # Deduplicate edges (same source/target pair, keep highest weight)
    all_edges_map: dict[tuple[str, str, str], Edge] = {}
    for edge in citation_edges + inheritance_edges:
        key = (edge.source, edge.target, edge.edge_type)
        if key not in all_edges_map or edge.weight > all_edges_map[key].weight:
            all_edges_map[key] = edge

    return LineageGraph(
        nodes=nodes,
        edges=list(all_edges_map.values()),
        threshold=threshold,
        n_pairs_evaluated=n_pairs,
    )


def ego_subgraph(graph: LineageGraph, paper_id: str, hops: int) -> LineageGraph:
    """Return the subgraph within `hops` edges of `paper_id`."""
    in_graph: set[str] = {paper_id}
    frontier: set[str] = {paper_id}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for edge in graph.edges:
            if edge.source in frontier and edge.target not in in_graph:
                next_frontier.add(edge.target)
            if edge.target in frontier and edge.source not in in_graph:
                next_frontier.add(edge.source)
        in_graph |= next_frontier
        frontier = next_frontier

    nodes = [n for n in graph.nodes if n.paper_id in in_graph]
    edges = [
        e for e in graph.edges
        if e.source in in_graph and e.target in in_graph
    ]
    return LineageGraph(
        nodes=nodes,
        edges=edges,
        threshold=graph.threshold,
        n_pairs_evaluated=graph.n_pairs_evaluated,
    )


# ============================================================
# 6. OUTPUT FORMATTERS
# ============================================================

# Colour palette per domain
_DOMAIN_COLOURS = {
    "AI":           "#6366F1",
    "ML":           "#8B5CF6",
    "Finance":      "#10B981",
    "Economics":    "#F59E0B",
    "Quantum":      "#3B82F6",
    "Biology":      "#EF4444",
    "Physics":      "#EC4899",
    "Neuroscience": "#14B8A6",
    "Unknown":      "#6B7280",
}


def format_html(graph: LineageGraph) -> str:
    """Generate a self-contained interactive HTML force-directed graph."""

    nodes_json = json.dumps([
        {
            "id": n.paper_id,
            "label": (n.title[:40] + "…") if len(n.title) > 40 else n.title,
            "title": n.title,
            "domain": n.domain,
            "confidence": n.overall_confidence,
            "n_modules": n.n_modules,
            "has_comparison": n.has_comparison,
            "color": _DOMAIN_COLOURS.get(n.domain, "#6B7280"),
            "size": max(10, int(n.overall_confidence * 24)),
        }
        for n in graph.nodes
    ], indent=2)

    edges_json = json.dumps([
        {
            "from": e.source,
            "to": e.target,
            "type": e.edge_type,
            "weight": e.weight,
            "label": e.label if e.edge_type == "inheritance" else "",
            "dashes": e.edge_type == "citation",
            "width": max(1, int(e.weight * 4)),
            "color": "#6366F1" if e.edge_type == "inheritance" else "#D1D5DB",
        }
        for e in graph.edges
    ], indent=2)

    domain_legend = "\n".join(
        f'<span class="legend-dot" style="background:{color}"></span>{domain}'
        for domain, color in _DOMAIN_COLOURS.items()
        if any(n.domain == domain for n in graph.nodes)
    )

    stats = (
        f"{len(graph.nodes)} papers · "
        f"{sum(1 for e in graph.edges if e.edge_type=='inheritance')} inheritance edges · "
        f"{sum(1 for e in graph.edges if e.edge_type=='citation')} citation edges · "
        f"threshold={graph.threshold}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArXivist SIR Lineage Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0F172A; color: #E2E8F0; height: 100vh; display: flex; flex-direction: column; }}
  header {{ padding: 12px 20px; background: #1E293B; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 16px; font-weight: 600; color: #F8FAFC; }}
  .stats {{ font-size: 12px; color: #94A3B8; }}
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin-left: auto; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  #graph {{ flex: 1; }}
  #info-panel {{
    position: absolute; top: 60px; right: 16px;
    width: 280px; background: #1E293B; border: 1px solid #334155;
    border-radius: 8px; padding: 14px; font-size: 13px;
    display: none; z-index: 10;
  }}
  #info-panel h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #F8FAFC; }}
  .info-row {{ display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #1E293B; }}
  .info-label {{ color: #94A3B8; }}
  .info-value {{ color: #E2E8F0; font-weight: 500; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #6366F1; color: white; }}
</style>
</head>
<body>
<header>
  <h1>ArXivist — SIR Lineage Graph</h1>
  <span class="stats">{stats}</span>
  <div class="legend" style="font-size:12px;">{domain_legend}</div>
</header>
<div id="graph"></div>
<div id="info-panel">
  <h3 id="info-title">—</h3>
  <div id="info-body"></div>
</div>
<script>
const nodesData = {nodes_json};
const edgesData = {edges_json};

const container = document.getElementById("graph");
const nodes = new vis.DataSet(nodesData.map(n => ({{
  id: n.id,
  label: n.label,
  title: `<b>${{n.title}}</b><br>Domain: ${{n.domain}}<br>Confidence: ${{n.confidence}}<br>Modules: ${{n.n_modules}}`,
  color: {{ background: n.color, border: n.color, highlight: {{ background: "#F8FAFC", border: n.color }} }},
  size: n.size,
  font: {{ color: "#F8FAFC", size: 11 }},
  _data: n,
}})));

const edges = new vis.DataSet(edgesData.map((e, i) => ({{
  id: i,
  from: e.from,
  to: e.to,
  label: e.label,
  dashes: e.dashes,
  width: e.width,
  color: {{ color: e.color, highlight: "#F8FAFC" }},
  arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
  font: {{ color: "#94A3B8", size: 10, align: "middle" }},
}})));

const network = new vis.Network(container, {{ nodes, edges }}, {{
  physics: {{
    forceAtlas2Based: {{ gravitationalConstant: -60, springLength: 120, springConstant: 0.08 }},
    solver: "forceAtlas2Based",
    stabilization: {{ iterations: 150 }},
  }},
  interaction: {{ hover: true, tooltipDelay: 100 }},
}});

network.on("click", function(params) {{
  const panel = document.getElementById("info-panel");
  if (params.nodes.length === 0) {{ panel.style.display = "none"; return; }}
  const node = nodes.get(params.nodes[0])._data;
  document.getElementById("info-title").textContent = node.title;
  document.getElementById("info-body").innerHTML = `
    <div class="info-row"><span class="info-label">Paper ID</span><span class="info-value">${{node.id}}</span></div>
    <div class="info-row"><span class="info-label">Domain</span><span class="info-value">${{node.domain}}</span></div>
    <div class="info-row"><span class="info-label">SIR Confidence</span><span class="info-value">${{node.confidence}}</span></div>
    <div class="info-row"><span class="info-label">Modules</span><span class="info-value">${{node.n_modules}}</span></div>
    ${{node.has_comparison ? '<div class="info-row"><span class="badge">✓ Reproduced</span></div>' : ""}}
  `;
  panel.style.display = "block";
  network.selectNodes([node.id]);
  network.focus(node.id, {{ scale: 1.2, animation: {{ duration: 400 }} }});
}});
</script>
</body>
</html>"""


def format_json_graph(graph: LineageGraph) -> str:
    return json.dumps(
        {
            "meta": {
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
                "threshold": graph.threshold,
                "n_pairs_evaluated": graph.n_pairs_evaluated,
                "n_inheritance_edges": sum(
                    1 for e in graph.edges if e.edge_type == "inheritance"
                ),
                "n_citation_edges": sum(
                    1 for e in graph.edges if e.edge_type == "citation"
                ),
            },
            "nodes": [
                {
                    "paper_id": n.paper_id,
                    "title": n.title,
                    "domain": n.domain,
                    "overall_confidence": n.overall_confidence,
                    "n_modules": n.n_modules,
                    "primary_metric": n.primary_metric,
                    "has_comparison": n.has_comparison,
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "label": e.label,
                }
                for e in graph.edges
            ],
        },
        indent=2,
    )


def format_dot(graph: LineageGraph) -> str:
    lines = [
        "digraph arxivist_lineage {",
        '  rankdir=LR;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontsize=9];',
        "",
    ]
    for node in graph.nodes:
        colour = _DOMAIN_COLOURS.get(node.domain, "#6B7280").lstrip("#")
        label = node.title.replace('"', '\\"')[:50]
        lines.append(
            f'  "{node.paper_id}" [label="{label}", fillcolor="#{colour}", fontcolor="white"];'
        )
    lines.append("")
    for edge in graph.edges:
        style = "solid" if edge.edge_type == "inheritance" else "dashed"
        label = edge.label.replace('"', '\\"')
        lines.append(
            f'  "{edge.source}" -> "{edge.target}" '
            f'[style={style}, label="{label}", weight={edge.weight}];'
        )
    lines.append("}")
    return "\n".join(lines)


def format_report(graph: LineageGraph) -> str:
    from collections import Counter

    degree: Counter = Counter()
    for edge in graph.edges:
        degree[edge.source] += 1
        degree[edge.target] += 1

    node_map = {n.paper_id: n for n in graph.nodes}
    isolated = [n for n in graph.nodes if degree[n.paper_id] == 0]
    top_connected = sorted(graph.nodes, key=lambda n: degree[n.paper_id], reverse=True)[:5]

    inheritance_edges = sorted(
        [e for e in graph.edges if e.edge_type == "inheritance"],
        key=lambda e: e.weight,
        reverse=True,
    )

    lines = [
        "# SIR Lineage Report",
        "",
        "## Summary",
        "",
        f"- Papers in registry: **{len(graph.nodes)}**",
        f"- Inheritance edges (similarity ≥ {graph.threshold}): "
        f"**{sum(1 for e in graph.edges if e.edge_type=='inheritance')}**",
        f"- Citation edges: **{sum(1 for e in graph.edges if e.edge_type=='citation')}**",
        f"- Isolated papers (no edges): **{len(isolated)}**",
        f"- Pairs evaluated: {graph.n_pairs_evaluated}",
        "",
        "## Most connected papers",
        "",
    ]
    for node in top_connected:
        d = degree[node.paper_id]
        lines.append(f"- **{node.title[:60]}** (`{node.paper_id}`) — {d} edge(s)")

    lines += [
        "",
        "## Strongest inheritance relationships",
        "",
    ]
    for edge in inheritance_edges[:10]:
        na = node_map.get(edge.source)
        nb = node_map.get(edge.target)
        title_a = na.title[:45] if na else edge.source
        title_b = nb.title[:45] if nb else edge.target
        lines.append(
            f"- `{edge.weight:.3f}` — **{title_a}** → **{title_b}**"
        )

    if isolated:
        lines += ["", "## Isolated papers (no relationships detected)", ""]
        for node in isolated:
            lines.append(f"- {node.title[:70]} (`{node.paper_id}`)")

    return "\n".join(lines)


# ============================================================
# 7. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ArXivist SIR Lineage Graph builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--registry-dir",
        default="../../workspace/sir-registry/",
        help="Path to the SIR registry directory",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Minimum SIR similarity score to draw an inheritance edge (default: 0.45)",
    )
    p.add_argument(
        "--domain",
        metavar="DOMAIN",
        default=None,
        help="Filter to a single subject domain (e.g. Finance, AI, Biology)",
    )
    p.add_argument(
        "--ego",
        metavar="PAPER_ID",
        default=None,
        help="Build ego graph centred on this paper_id",
    )
    p.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Number of hops for ego graph (default: 2)",
    )
    p.add_argument(
        "--format",
        choices=["html", "json", "dot", "report"],
        default="html",
        help="Output format (default: html)",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Write output to file (default: stdout for report/json/dot, lineage.html for html)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading registry from '{args.registry_dir}'...", file=sys.stderr)
    sirs = load_registry(args.registry_dir)

    if not sirs:
        print(
            f"No SIRs found in '{args.registry_dir}'.\n"
            "Run the ArXivist pipeline first to populate the registry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building lineage graph for {len(sirs)} SIRs...", file=sys.stderr)
    graph = build_graph(sirs, threshold=args.threshold, domain_filter=args.domain)

    if args.ego:
        graph = ego_subgraph(graph, args.ego, args.hops)
        print(
            f"Ego graph ({args.hops} hops from {args.ego}): "
            f"{len(graph.nodes)} nodes, {len(graph.edges)} edges",
            file=sys.stderr,
        )

    print(
        f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges "
        f"({sum(1 for e in graph.edges if e.edge_type=='inheritance')} inheritance, "
        f"{sum(1 for e in graph.edges if e.edge_type=='citation')} citation)",
        file=sys.stderr,
    )

    if args.format == "html":
        output = format_html(graph)
        out_path = args.out or "lineage.html"
        Path(out_path).write_text(output, encoding="utf-8")
        print(f"Interactive graph written to: {out_path}", file=sys.stderr)
        return
    elif args.format == "json":
        output = format_json_graph(graph)
    elif args.format == "dot":
        output = format_dot(graph)
    else:
        output = format_report(graph)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Output written to: {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

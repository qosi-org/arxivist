#!/usr/bin/env python3
"""
sir_diff.py  —  ArXivist SIR Diff Engine
==========================================

Compare two SIR JSON files and produce a structured diff report with a formal
similarity score. Useful for:

  - Comparing a paper against its follow-up work
  - Measuring how much a new paper inherits from a prior architecture
  - Tracking what changed between two ArXivist runs on revised paper versions
  - Identifying which architectural decisions differentiate competing papers

SIMILARITY SCORE
----------------
The overall SIR similarity score (0.0–1.0) is a weighted average across sections:

  Section                Weight
  ─────────────────────────────
  architecture           0.35    (module overlap, connection overlap)
  mathematical_spec      0.20    (equation name overlap)
  training_pipeline      0.20    (optimizer, schedule, batch size match)
  evaluation_protocol    0.15    (metric and dataset overlap)
  tensor_semantics       0.10    (shape notation overlap)

A score ≥ 0.85 means the papers share nearly identical implementations.
A score 0.60–0.84 means significant architectural inheritance.
A score 0.30–0.59 means same domain/paradigm, different design.
A score < 0.30 means largely independent implementations.

OUTPUT FORMATS
--------------
  --format markdown   Human-readable report (default)
  --format json       Machine-readable diff object
  --format summary    One-line similarity score only

USAGE
-----
  # Compare two SIR files directly
  python sir_diff.py sir_a.json sir_b.json

  # Compare two papers in the registry by paper_id
  python sir_diff.py --registry-dir ../../workspace/sir-registry/ \\
      --id-a arxiv_1706_03762 --id-b arxiv_2005_14165

  # Output as JSON
  python sir_diff.py sir_a.json sir_b.json --format json

  # One-line summary only
  python sir_diff.py sir_a.json sir_b.json --format summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# 1. DATA STRUCTURES
# ============================================================

@dataclass
class SectionDiff:
    """Diff result for a single SIR section."""
    section: str
    weight: float
    similarity: float          # 0.0–1.0 for this section
    added: list[str]           # items in B not in A
    removed: list[str]         # items in A not in B
    changed: list[tuple[str, Any, Any]]   # (field, value_a, value_b)
    unchanged: list[str]       # items identical in A and B


@dataclass
class SIRDiff:
    """Complete diff result between two SIRs."""
    paper_id_a: str
    paper_id_b: str
    title_a: str
    title_b: str
    overall_similarity: float
    interpretation: str
    section_diffs: list[SectionDiff]
    domain_a: str
    domain_b: str
    domain_match: bool


# ============================================================
# 2. SECTION-LEVEL SIMILARITY FUNCTIONS
# ============================================================

def _name_set(items: list[dict], key: str = "name") -> set[str]:
    """Extract a set of name strings from a list of dicts."""
    return {str(item.get(key, "")).lower() for item in items if item.get(key)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _value_similarity(a: Any, b: Any) -> float:
    """Fuzzy similarity between two scalar values."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0 and b == 0:
            return 1.0
        if a == 0 or b == 0:
            return 0.0
        ratio = min(a, b) / max(a, b)
        return ratio
    # String comparison
    sa, sb = str(a).lower().strip(), str(b).lower().strip()
    if sa == sb:
        return 1.0
    # Partial match
    if sa in sb or sb in sa:
        return 0.7
    # Token overlap
    ta = set(re.findall(r"[a-z0-9]+", sa))
    tb = set(re.findall(r"[a-z0-9]+", sb))
    return _jaccard(ta, tb)


def diff_architecture(arch_a: dict, arch_b: dict) -> SectionDiff:
    """Diff the architecture sections of two SIRs."""
    mods_a = arch_a.get("modules", [])
    mods_b = arch_b.get("modules", [])
    names_a = _name_set(mods_a)
    names_b = _name_set(mods_b)

    conn_a = {
        f"{c.get('from','')}→{c.get('to','')}"
        for c in arch_a.get("connections", [])
    }
    conn_b = {
        f"{c.get('from','')}→{c.get('to','')}"
        for c in arch_b.get("connections", [])
    }

    # Variant comparison
    var_a = {v.get("name", "").lower() for v in arch_a.get("variants", [])}
    var_b = {v.get("name", "").lower() for v in arch_b.get("variants", [])}

    module_sim = _jaccard(names_a, names_b)
    conn_sim = _jaccard(conn_a, conn_b)
    variant_sim = _jaccard(var_a, var_b) if (var_a or var_b) else 1.0

    similarity = 0.60 * module_sim + 0.25 * conn_sim + 0.15 * variant_sim

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)
    unchanged = sorted(names_a & names_b)

    changed = []
    # Compare operation types for shared modules
    map_a = {m.get("name", "").lower(): m for m in mods_a}
    map_b = {m.get("name", "").lower(): m for m in mods_b}
    for name in names_a & names_b:
        op_a = map_a[name].get("operation_type", "")
        op_b = map_b[name].get("operation_type", "")
        if op_a and op_b and op_a.lower() != op_b.lower():
            changed.append((f"module:{name}:operation_type", op_a, op_b))

    return SectionDiff(
        section="architecture",
        weight=0.35,
        similarity=similarity,
        added=[f"module:{n}" for n in added],
        removed=[f"module:{n}" for n in removed],
        changed=changed,
        unchanged=[f"module:{n}" for n in unchanged],
    )


def diff_mathematical_spec(spec_a: list, spec_b: list) -> SectionDiff:
    """Diff the mathematical specification sections."""
    names_a = {e.get("name", "").lower() for e in spec_a if e.get("name")}
    names_b = {e.get("name", "").lower() for e in spec_b if e.get("name")}

    # Role overlap
    roles_a = {e.get("role", "").lower() for e in spec_a if e.get("role")}
    roles_b = {e.get("role", "").lower() for e in spec_b if e.get("role")}

    name_sim = _jaccard(names_a, names_b)
    role_sim = _jaccard(roles_a, roles_b)
    similarity = 0.7 * name_sim + 0.3 * role_sim

    # Check for changed roles in shared equations
    changed = []
    map_a = {e.get("name", "").lower(): e for e in spec_a if e.get("name")}
    map_b = {e.get("name", "").lower(): e for e in spec_b if e.get("name")}
    for name in names_a & names_b:
        role_a = map_a[name].get("role", "")
        role_b = map_b[name].get("role", "")
        if role_a and role_b and role_a != role_b:
            changed.append((f"equation:{name}:role", role_a, role_b))

    return SectionDiff(
        section="mathematical_spec",
        weight=0.20,
        similarity=similarity,
        added=sorted(names_b - names_a),
        removed=sorted(names_a - names_b),
        changed=changed,
        unchanged=sorted(names_a & names_b),
    )


def diff_training_pipeline(tp_a: dict, tp_b: dict) -> SectionDiff:
    """Diff the training pipeline sections."""
    fields_to_compare = [
        ("optimizer.name",    tp_a.get("optimizer", {}).get("name"),
                              tp_b.get("optimizer", {}).get("name")),
        ("optimizer.lr",      tp_a.get("optimizer", {}).get("learning_rate"),
                              tp_b.get("optimizer", {}).get("learning_rate")),
        ("optimizer.beta1",   tp_a.get("optimizer", {}).get("beta1"),
                              tp_b.get("optimizer", {}).get("beta1")),
        ("optimizer.beta2",   tp_a.get("optimizer", {}).get("beta2"),
                              tp_b.get("optimizer", {}).get("beta2")),
        ("lr_schedule.type",  tp_a.get("lr_schedule", {}).get("type"),
                              tp_b.get("lr_schedule", {}).get("type")),
        ("warmup_steps",      tp_a.get("lr_schedule", {}).get("warmup_steps"),
                              tp_b.get("lr_schedule", {}).get("warmup_steps")),
        ("batch_size",        tp_a.get("batch_size"),       tp_b.get("batch_size")),
        ("mixed_precision",   tp_a.get("mixed_precision"),  tp_b.get("mixed_precision")),
        ("gradient_clipping", tp_a.get("gradient_clipping"), tp_b.get("gradient_clipping")),
    ]

    field_sims = []
    changed = []
    unchanged = []

    for fname, val_a, val_b in fields_to_compare:
        if val_a is None and val_b is None:
            continue
        sim = _value_similarity(val_a, val_b)
        field_sims.append(sim)
        if sim < 0.95:
            changed.append((fname, val_a, val_b))
        else:
            unchanged.append(fname)

    similarity = sum(field_sims) / len(field_sims) if field_sims else 1.0

    return SectionDiff(
        section="training_pipeline",
        weight=0.20,
        similarity=similarity,
        added=[],
        removed=[],
        changed=changed,
        unchanged=unchanged,
    )


def diff_evaluation_protocol(ep_a: dict, ep_b: dict) -> SectionDiff:
    """Diff the evaluation protocol sections."""
    metrics_a = {m.lower() for m in ep_a.get("metrics", [])}
    metrics_b = {m.lower() for m in ep_b.get("metrics", [])}

    ds_a = {d.get("name", "").lower() for d in ep_a.get("datasets", []) if d.get("name")}
    ds_b = {d.get("name", "").lower() for d in ep_b.get("datasets", []) if d.get("name")}

    metric_sim = _jaccard(metrics_a, metrics_b)
    dataset_sim = _jaccard(ds_a, ds_b)
    similarity = 0.6 * metric_sim + 0.4 * dataset_sim

    added = sorted((metrics_b - metrics_a) | (ds_b - ds_a))
    removed = sorted((metrics_a - metrics_b) | (ds_a - ds_b))
    unchanged = sorted((metrics_a & metrics_b) | (ds_a & ds_b))

    return SectionDiff(
        section="evaluation_protocol",
        weight=0.15,
        similarity=similarity,
        added=added,
        removed=removed,
        changed=[],
        unchanged=unchanged,
    )


def diff_tensor_semantics(ts_a: list, ts_b: list) -> SectionDiff:
    """Diff tensor semantics sections."""
    names_a = _name_set(ts_a)
    names_b = _name_set(ts_b)

    # Shape notation overlap for shared tensors
    map_a = {t.get("name", "").lower(): t for t in ts_a if t.get("name")}
    map_b = {t.get("name", "").lower(): t for t in ts_b if t.get("name")}

    changed = []
    for name in names_a & names_b:
        shape_a = map_a[name].get("shape_notation", "")
        shape_b = map_b[name].get("shape_notation", "")
        if shape_a and shape_b and shape_a != shape_b:
            changed.append((f"tensor:{name}:shape", shape_a, shape_b))

    similarity = _jaccard(names_a, names_b)

    return SectionDiff(
        section="tensor_semantics",
        weight=0.10,
        similarity=similarity,
        added=sorted(names_b - names_a),
        removed=sorted(names_a - names_b),
        changed=changed,
        unchanged=sorted(names_a & names_b),
    )


# ============================================================
# 3. OVERALL DIFF COMPUTATION
# ============================================================

def _interpret(score: float) -> str:
    if score >= 0.85:
        return "Nearly identical implementations — likely the same architecture or a direct extension."
    if score >= 0.60:
        return "Significant architectural inheritance — one paper likely builds on the other."
    if score >= 0.30:
        return "Same paradigm / domain, but substantially different design choices."
    return "Largely independent implementations — low overlap across all sections."


def compute_diff(sir_a: dict, sir_b: dict) -> SIRDiff:
    """Compute the full diff between two SIRs."""
    section_diffs = [
        diff_architecture(
            sir_a.get("architecture", {}),
            sir_b.get("architecture", {}),
        ),
        diff_mathematical_spec(
            sir_a.get("mathematical_spec", []),
            sir_b.get("mathematical_spec", []),
        ),
        diff_training_pipeline(
            sir_a.get("training_pipeline", {}),
            sir_b.get("training_pipeline", {}),
        ),
        diff_evaluation_protocol(
            sir_a.get("evaluation_protocol", {}),
            sir_b.get("evaluation_protocol", {}),
        ),
        diff_tensor_semantics(
            sir_a.get("tensor_semantics", []),
            sir_b.get("tensor_semantics", []),
        ),
    ]

    # Weighted average
    total_weight = sum(sd.weight for sd in section_diffs)
    overall = sum(sd.similarity * sd.weight for sd in section_diffs) / total_weight

    prov_a = sir_a.get("provenance", {})
    prov_b = sir_b.get("provenance", {})
    domain_a = prov_a.get("subject_domain") or prov_a.get("domain", "?")
    domain_b = prov_b.get("subject_domain") or prov_b.get("domain", "?")

    return SIRDiff(
        paper_id_a=sir_a.get("paper_id", "?"),
        paper_id_b=sir_b.get("paper_id", "?"),
        title_a=prov_a.get("title", sir_a.get("paper_id", "?")),
        title_b=prov_b.get("title", sir_b.get("paper_id", "?")),
        overall_similarity=round(overall, 4),
        interpretation=_interpret(overall),
        section_diffs=section_diffs,
        domain_a=str(domain_a),
        domain_b=str(domain_b),
        domain_match=str(domain_a).lower() == str(domain_b).lower(),
    )


# ============================================================
# 4. OUTPUT FORMATTERS
# ============================================================

def format_markdown(diff: SIRDiff) -> str:
    lines = [
        "# SIR Diff Report",
        "",
        f"**Paper A**: {diff.title_a}  (`{diff.paper_id_a}`)",
        f"**Paper B**: {diff.title_b}  (`{diff.paper_id_b}`)",
        f"**Domain A**: {diff.domain_a}  |  **Domain B**: {diff.domain_b}"
        + ("  ✓ same domain" if diff.domain_match else "  ✗ different domains"),
        "",
        "## Overall Similarity",
        "",
        f"**Score**: `{diff.overall_similarity:.4f}` / 1.0",
        "",
        f"> {diff.interpretation}",
        "",
        "## Section Breakdown",
        "",
        "| Section | Weight | Similarity | Added | Removed | Changed |",
        "|---|---|---|---|---|---|",
    ]

    for sd in diff.section_diffs:
        bar = "█" * int(sd.similarity * 10) + "░" * (10 - int(sd.similarity * 10))
        lines.append(
            f"| {sd.section} | {sd.weight:.2f} | {sd.similarity:.3f} `{bar}` "
            f"| {len(sd.added)} | {len(sd.removed)} | {len(sd.changed)} |"
        )

    lines += ["", "## Details by Section", ""]

    for sd in diff.section_diffs:
        lines.append(f"### {sd.section.replace('_', ' ').title()}")
        lines.append(f"Similarity: **{sd.similarity:.3f}**")
        lines.append("")

        if sd.added:
            lines.append(f"**Added in B** ({len(sd.added)}):")
            for item in sd.added[:10]:
                lines.append(f"  + {item}")
            if len(sd.added) > 10:
                lines.append(f"  ... and {len(sd.added) - 10} more")
            lines.append("")

        if sd.removed:
            lines.append(f"**Removed from A** ({len(sd.removed)}):")
            for item in sd.removed[:10]:
                lines.append(f"  - {item}")
            if len(sd.removed) > 10:
                lines.append(f"  ... and {len(sd.removed) - 10} more")
            lines.append("")

        if sd.changed:
            lines.append(f"**Changed** ({len(sd.changed)}):")
            for fname, va, vb in sd.changed[:8]:
                lines.append(f"  ~ {fname}: `{va}` → `{vb}`")
            if len(sd.changed) > 8:
                lines.append(f"  ... and {len(sd.changed) - 8} more")
            lines.append("")

        if sd.unchanged:
            lines.append(
                f"**Unchanged**: {', '.join(sd.unchanged[:6])}"
                + (f" ... (+{len(sd.unchanged)-6} more)" if len(sd.unchanged) > 6 else "")
            )
            lines.append("")

    return "\n".join(lines)


def format_json(diff: SIRDiff) -> str:
    def sd_to_dict(sd: SectionDiff) -> dict:
        return {
            "section": sd.section,
            "weight": sd.weight,
            "similarity": sd.similarity,
            "added": sd.added,
            "removed": sd.removed,
            "changed": [
                {"field": f, "value_a": va, "value_b": vb}
                for f, va, vb in sd.changed
            ],
            "unchanged_count": len(sd.unchanged),
        }

    return json.dumps(
        {
            "paper_id_a": diff.paper_id_a,
            "paper_id_b": diff.paper_id_b,
            "title_a": diff.title_a,
            "title_b": diff.title_b,
            "domain_a": diff.domain_a,
            "domain_b": diff.domain_b,
            "domain_match": diff.domain_match,
            "overall_similarity": diff.overall_similarity,
            "interpretation": diff.interpretation,
            "sections": [sd_to_dict(sd) for sd in diff.section_diffs],
        },
        indent=2,
    )


def format_summary(diff: SIRDiff) -> str:
    return (
        f"Similarity: {diff.overall_similarity:.4f}  |  "
        f"{diff.paper_id_a} vs {diff.paper_id_b}  |  "
        f"{diff.interpretation}"
    )


# ============================================================
# 5. CLI
# ============================================================

def load_sir(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_from_registry(registry_dir: str, paper_id: str) -> dict:
    registry_path = Path(registry_dir)
    candidates = list(registry_path.rglob(f"{paper_id}/sir.json"))
    if not candidates:
        print(f"Error: No SIR found for paper_id='{paper_id}' in '{registry_dir}'", file=sys.stderr)
        sys.exit(1)
    with open(candidates[0]) as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ArXivist SIR Diff Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("sir_a", nargs="?", help="Path to first SIR JSON file")
    p.add_argument("sir_b", nargs="?", help="Path to second SIR JSON file")
    p.add_argument(
        "--registry-dir",
        default="../../workspace/sir-registry/",
        help="Registry directory (for --id-a / --id-b mode)",
    )
    p.add_argument("--id-a", metavar="PAPER_ID", help="Paper ID for first SIR")
    p.add_argument("--id-b", metavar="PAPER_ID", help="Paper ID for second SIR")
    p.add_argument(
        "--format",
        choices=["markdown", "json", "summary"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        help="Write output to file instead of stdout",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load SIRs
    if args.id_a and args.id_b:
        sir_a = load_from_registry(args.registry_dir, args.id_a)
        sir_b = load_from_registry(args.registry_dir, args.id_b)
    elif args.sir_a and args.sir_b:
        sir_a = load_sir(args.sir_a)
        sir_b = load_sir(args.sir_b)
    else:
        parser.error(
            "Provide either two SIR file paths (sir_a sir_b) "
            "or --id-a and --id-b with --registry-dir"
        )

    diff = compute_diff(sir_a, sir_b)

    if args.format == "markdown":
        output = format_markdown(diff)
    elif args.format == "json":
        output = format_json(diff)
    else:
        output = format_summary(diff)

    if args.out:
        Path(args.out).write_text(output)
        print(f"Diff written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

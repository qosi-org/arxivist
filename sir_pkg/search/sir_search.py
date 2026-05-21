#!/usr/bin/env python3
"""
sir_search.py  —  ArXivist SIR Registry Search
================================================

Query the ArXivist SIR registry using natural language or structured filters.
Finds papers by architecture detail, training configuration, metric, domain,
or any other field stored in the SIR.

No embedding model or API required by default — uses TF-IDF over SIR text fields
plus structured field matching for precise queries. If sentence-transformers is
installed, semantic search is automatically used for natural language queries.

QUERY MODES
-----------
  Natural language  (default):
      "papers that use cross-attention with RoPE positional encoding"
      "transformer models trained with AdamW and cosine schedule"

  Structured filters  (--filter):
      domain=AI
      domain=Finance AND metric=Sharpe
      module=MultiHeadAttention AND confidence.architecture>0.9
      optimizer=Adam AND batch_size<64

  Field lookup  (--field):
      Show the value of a specific field across all SIRs:
      python sir_search.py --field training_pipeline.optimizer.name

  Paper lookup  (--paper-id):
      Retrieve and display a specific SIR by paper ID.

USAGE
-----
  # Natural language search
  python sir_search.py "papers using diffusion with DDPM noise schedule"

  # Structured filter search
  python sir_search.py --filter "domain=Finance AND metric=Sharpe"

  # Show specific field across all papers
  python sir_search.py --field training_pipeline.optimizer.name

  # Look up a specific paper
  python sir_search.py --paper-id arxiv_1706_03762

  # Output as JSON (for piping)
  python sir_search.py "attention mechanism" --json

  # Top-N results
  python sir_search.py "variational autoencoder" --top 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# 1. REGISTRY LOADING
# ============================================================

def load_registry(registry_dir: str) -> list[dict]:
    """Load all canonical SIRs from the registry directory."""
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
        sirs.append(sir)

    return sirs


# ============================================================
# 2. SIR → SEARCHABLE TEXT
# ============================================================

def sir_to_text(sir: dict) -> str:
    """Flatten a SIR into a single searchable text string.

    Extracts all human-meaningful text fields: title, abstract, key claims,
    module names and operation types, optimizer, metric names, equation names,
    assumption descriptions, ambiguity descriptions, domain.
    """
    parts: list[str] = []

    prov = sir.get("provenance", {})
    parts.append(prov.get("title", ""))
    parts.append(prov.get("abstract", ""))
    parts.append(prov.get("domain", ""))
    parts.append(prov.get("subject_domain", "") or "")
    parts.extend(prov.get("key_claims", []))
    parts.extend(prov.get("authors", []))

    arch = sir.get("architecture", {})
    for module in arch.get("modules", []):
        parts.append(module.get("name", ""))
        parts.append(module.get("operation_type", ""))
        parts.append(module.get("notes", "") or "")

    for eq in sir.get("mathematical_spec", []):
        parts.append(eq.get("name", ""))
        parts.append(eq.get("role", ""))

    for tensor in sir.get("tensor_semantics", []):
        parts.append(tensor.get("name", ""))

    tp = sir.get("training_pipeline", {})
    opt = tp.get("optimizer", {})
    if isinstance(opt, dict):
        parts.append(opt.get("name", "") or "")
    sched = tp.get("lr_schedule", {})
    if isinstance(sched, dict):
        parts.append(sched.get("type", "") or "")
    parts.append(tp.get("mixed_precision", "") or "")

    ep = sir.get("evaluation_protocol", {})
    parts.extend(ep.get("metrics", []))
    for ds in ep.get("datasets", []):
        parts.append(ds.get("name", ""))
    for result in ep.get("reported_results", []):
        parts.append(result.get("metric", ""))
        parts.append(result.get("dataset", ""))

    for assumption in sir.get("implementation_assumptions", []):
        parts.append(assumption.get("assumption", ""))
        parts.append(assumption.get("basis", ""))

    for amb in sir.get("ambiguities", []):
        parts.append(amb.get("description", ""))

    return " ".join(p for p in parts if p and isinstance(p, str))


# ============================================================
# 3. TF-IDF ENGINE
# ============================================================

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFIndex:
    """Lightweight TF-IDF index over SIR text representations."""

    def __init__(self):
        self.sirs: list[dict] = []
        self.doc_texts: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.tf_vecs: list[dict[str, float]] = []

    def build(self, sirs: list[dict]) -> None:
        self.sirs = sirs
        self.doc_texts = [sir_to_text(s) for s in sirs]
        self.doc_tokens = [tokenize(t) for t in self.doc_texts]
        n_docs = len(self.doc_tokens)

        # IDF
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] += 1
        self.idf = {
            t: math.log((n_docs + 1) / (freq + 1)) + 1
            for t, freq in df.items()
        }

        # TF vectors
        self.tf_vecs = []
        for tokens in self.doc_tokens:
            total = len(tokens) or 1
            tf: dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1.0 / total
            self.tf_vecs.append(dict(tf))

    def query(self, query_text: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """Return top-k SIRs matching the query, with scores."""
        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []

        scores = []
        for idx, (tf_vec, sir) in enumerate(zip(self.tf_vecs, self.sirs)):
            score = 0.0
            for t in q_tokens:
                idf = self.idf.get(t, 0.0)
                tf = tf_vec.get(t, 0.0)
                score += tf * idf
            scores.append((sir, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(s, sc) for s, sc in scores[:top_k] if sc > 0]


# ============================================================
# 4. SEMANTIC SEARCH (optional — requires sentence-transformers)
# ============================================================

def _try_semantic(
    sirs: list[dict],
    query: str,
    top_k: int,
) -> list[tuple[dict, float]] | None:
    """Attempt semantic search using sentence-transformers.
    Returns None if the library is not installed."""
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_texts = [sir_to_text(s) for s in sirs]
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(top_k, len(sirs)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append((sirs[int(idx)], float(score)))
    return results


# ============================================================
# 5. STRUCTURED FILTER ENGINE
# ============================================================

def _nested_get(obj: Any, dotpath: str, default=None) -> Any:
    """Traverse a nested dict/list using dot notation.
    e.g. "training_pipeline.optimizer.name"
    """
    parts = dotpath.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, default)
        elif isinstance(current, list):
            # For lists, collect all matching values
            results = []
            for item in current:
                if isinstance(item, dict):
                    val = item.get(part, default)
                    if val is not None:
                        results.append(val)
            return results if results else default
        else:
            return default
    return current


def _parse_filter(filter_str: str) -> list[tuple[str, str, str]]:
    """Parse a filter expression into (field, operator, value) triples.

    Supports:
        field=value          exact match (string)
        field!=value         not equal
        field>value          greater than (numeric)
        field<value          less than (numeric)
        field~value          contains (substring, case-insensitive)

    Multiple conditions joined with AND.
    """
    conditions = []
    for part in re.split(r"\s+AND\s+", filter_str, flags=re.IGNORECASE):
        part = part.strip()
        for op in ("!=", ">=", "<=", ">", "<", "~", "="):
            if op in part:
                field, _, value = part.partition(op)
                conditions.append((field.strip(), op, value.strip()))
                break
    return conditions


def _apply_condition(sir: dict, field: str, op: str, value: str) -> bool:
    """Test whether a SIR satisfies a single filter condition."""
    # Shorthand field aliases
    aliases = {
        "domain":     "provenance.subject_domain",
        "title":      "provenance.title",
        "optimizer":  "training_pipeline.optimizer.name",
        "metric":     "evaluation_protocol.metrics",
        "module":     "architecture.modules",
        "confidence": "confidence_annotations.overall_sir_confidence",
        "batch_size": "training_pipeline.batch_size",
        "framework":  "architecture.primary_variant",
    }
    dotpath = aliases.get(field, field)
    raw = _nested_get(sir, dotpath)

    if raw is None:
        return False

    # List values: check if any element satisfies the condition
    if isinstance(raw, list):
        return any(_compare(str(item), op, value) for item in raw)

    return _compare(str(raw), op, value)


def _compare(raw: str, op: str, value: str) -> bool:
    if op == "=":
        return raw.lower() == value.lower()
    if op == "!=":
        return raw.lower() != value.lower()
    if op == "~":
        return value.lower() in raw.lower()
    # Numeric comparisons
    try:
        raw_num = float(raw)
        val_num = float(value)
        if op == ">":  return raw_num > val_num
        if op == "<":  return raw_num < val_num
        if op == ">=": return raw_num >= val_num
        if op == "<=": return raw_num <= val_num
    except (ValueError, TypeError):
        pass
    return False


def filter_sirs(sirs: list[dict], filter_str: str) -> list[dict]:
    """Return SIRs matching all conditions in the filter expression."""
    conditions = _parse_filter(filter_str)
    if not conditions:
        return sirs

    results = []
    for sir in sirs:
        if all(_apply_condition(sir, f, op, v) for f, op, v in conditions):
            results.append(sir)
    return results


# ============================================================
# 6. FIELD REPORT
# ============================================================

def field_report(sirs: list[dict], dotpath: str) -> list[dict]:
    """Collect the value of a specific field across all SIRs."""
    rows = []
    for sir in sirs:
        value = _nested_get(sir, dotpath)
        rows.append({
            "paper_id": sir.get("paper_id", "?"),
            "title": _nested_get(sir, "provenance.title", ""),
            "field": dotpath,
            "value": value,
        })
    return sorted(rows, key=lambda r: str(r["value"]))


# ============================================================
# 7. RESULT FORMATTING
# ============================================================

def format_result(sir: dict, score: float | None = None, verbose: bool = False) -> str:
    """Format a single SIR result for terminal display."""
    prov = sir.get("provenance", {})
    title = prov.get("title", sir.get("paper_id", "Unknown"))
    pid = sir.get("paper_id", "?")
    domain = prov.get("subject_domain") or prov.get("domain", "?")
    overall_conf = sir.get("confidence_annotations", {}).get("overall_sir_confidence", "?")

    arch = sir.get("architecture", {})
    modules = [m.get("name", "") for m in arch.get("modules", []) if m.get("name")]
    n_modules = len(modules)
    module_preview = ", ".join(modules[:4]) + ("..." if n_modules > 4 else "")

    tp = sir.get("training_pipeline", {})
    opt_name = tp.get("optimizer", {}).get("name", "?") if isinstance(tp.get("optimizer"), dict) else "?"

    ep = sir.get("evaluation_protocol", {})
    metrics = ", ".join(ep.get("metrics", [])[:3]) or "?"

    score_str = f"  score={score:.4f}" if score is not None else ""
    lines = [
        f"┌─ {title}",
        f"│  id={pid}  domain={domain}  confidence={overall_conf}{score_str}",
        f"│  modules ({n_modules}): {module_preview}",
        f"│  optimizer={opt_name}  metrics={metrics}",
    ]

    if verbose:
        abstract = prov.get("abstract", "")[:200]
        if abstract:
            lines.append(f"│  abstract: {abstract}...")
        assumptions = sir.get("implementation_assumptions", [])
        if assumptions:
            lines.append(f"│  assumptions ({len(assumptions)}): "
                         + assumptions[0].get("assumption", "")[:100])

    lines.append("└" + "─" * 60)
    return "\n".join(lines)


# ============================================================
# 8. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ArXivist SIR Search — query the SIR registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural language search query",
    )
    p.add_argument(
        "--registry-dir",
        default="../../workspace/sir-registry/",
        help="Path to the SIR registry directory",
    )
    p.add_argument(
        "--filter",
        metavar="EXPR",
        default=None,
        help='Structured filter expression e.g. "domain=Finance AND metric=Sharpe"',
    )
    p.add_argument(
        "--field",
        metavar="DOTPATH",
        default=None,
        help="Show a specific field across all SIRs e.g. training_pipeline.optimizer.name",
    )
    p.add_argument(
        "--paper-id",
        metavar="ID",
        default=None,
        help="Retrieve and display a specific SIR by paper_id",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show more detail per result",
    )
    p.add_argument(
        "--semantic",
        action="store_true",
        help="Force semantic search (requires sentence-transformers)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load registry
    sirs = load_registry(args.registry_dir)
    if not sirs:
        print(
            f"No SIRs found in '{args.registry_dir}'.\n"
            "Run the ArXivist pipeline first to populate the registry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Registry loaded: {len(sirs)} SIRs", file=sys.stderr)

    # --- Paper ID lookup ---
    if args.paper_id:
        matches = [s for s in sirs if s.get("paper_id") == args.paper_id]
        if not matches:
            print(f"No SIR found with paper_id='{args.paper_id}'", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(matches[0], indent=2))
        else:
            print(format_result(matches[0], verbose=True))
        return

    # --- Field report ---
    if args.field:
        rows = field_report(sirs, args.field)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"\nField: {args.field}  ({len(rows)} papers)\n")
            for row in rows:
                val_str = str(row["value"])[:80] if row["value"] is not None else "null"
                print(f"  {row['paper_id']:<30}  {val_str}")
        return

    # --- Structured filter ---
    if args.filter:
        results = filter_sirs(sirs, args.filter)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nFilter: {args.filter}")
            print(f"Results: {len(results)} / {len(sirs)} papers\n")
            for sir in results[: args.top]:
                print(format_result(sir, verbose=args.verbose))
        return

    # --- Natural language search ---
    if args.query:
        results = None

        # Try semantic search first if requested or available
        if args.semantic:
            results = _try_semantic(sirs, args.query, args.top)
            if results is None:
                print(
                    "sentence-transformers not installed. "
                    "Falling back to TF-IDF.\n"
                    "Install with: pip install sentence-transformers",
                    file=sys.stderr,
                )

        # TF-IDF fallback (or default)
        if results is None:
            index = TFIDFIndex()
            index.build(sirs)
            results = index.query(args.query, top_k=args.top)

        if not results:
            print("No results found.", file=sys.stderr)
            sys.exit(0)

        if args.json:
            print(json.dumps([s for s, _ in results], indent=2))
        else:
            print(f"\nQuery: \"{args.query}\"")
            print(f"Results: {len(results)} (top {args.top})\n")
            for sir, score in results:
                print(format_result(sir, score=score, verbose=args.verbose))
        return

    # No mode selected
    parser.print_help()


if __name__ == "__main__":
    main()

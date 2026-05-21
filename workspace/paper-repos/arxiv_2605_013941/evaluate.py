#!/usr/bin/env python3
"""
evaluate.py — Evaluate EVOLVEMEM on a QA benchmark.

Runs evaluation on a QA set with the current (or provided) retrieval config
and writes per-question results + aggregate metrics.

Usage:
    python evaluate.py --qa-file data/locomo_qa.jsonl \\
                       --db-path evolvemem.db \\
                       --config configs/config.yaml \\
                       --output-dir outputs/ \\
                       --benchmark locomo
"""

import argparse
import json
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="EVOLVEMEM: Evaluate on QA benchmark")
    parser.add_argument("--qa-file", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--evolved-config", default=None, help="Path to evolved config JSON (optional)")
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--benchmark", choices=["locomo", "membench"], default="locomo")
    parser.add_argument("--debug", action="store_true", help="Use first 20 QA pairs only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["memory"]["db_path"] = args.db_path

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    qa_pairs = []
    with open(args.qa_file) as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line.strip()))

    if args.debug:
        qa_pairs = qa_pairs[:20]

    logger.info(f"Evaluating {len(qa_pairs)} QA pairs on {args.benchmark}...")

    from src.evolvemem.evolvemem import EvolveMem
    from src.evolvemem.evaluation.metrics import Evaluator

    em = EvolveMem(config_path=tmp_config)

    if args.evolved_config:
        em.load_state(args.evolved_config)
        logger.info(f"Loaded evolved config from {args.evolved_config}")

    evaluator = Evaluator()
    results = []
    use_f1 = args.benchmark == "locomo"

    for item in qa_pairs:
        query = item.get("q", item.get("question", ""))
        reference = item.get("ref", item.get("reference", item.get("answer", "")))
        category = item.get("category")

        prediction = em.answer(query, category=category)

        if use_f1:
            score = evaluator.token_f1(prediction, reference)
            bleu = evaluator.bleu1(prediction, reference)
            results.append({"q": query, "pred": prediction, "ref": reference,
                            "f1": score, "bleu": bleu, "category": category})
        else:
            score = evaluator.exact_match(prediction, reference)
            results.append({"q": query, "pred": prediction, "ref": reference,
                            "em": score, "category": category})

    # Aggregate
    if use_f1:
        overall_f1 = sum(r["f1"] for r in results) / len(results)
        overall_bleu = sum(r["bleu"] for r in results) / len(results)
        logger.info(f"Overall F1: {overall_f1:.4f} | BLEU-1: {overall_bleu:.4f}")
        summary = {"overall_f1": overall_f1, "overall_bleu1": overall_bleu, "n": len(results)}
    else:
        overall_acc = sum(r["em"] for r in results) / len(results)
        logger.info(f"Overall Accuracy: {overall_acc:.4f}")
        summary = {"overall_accuracy": overall_acc, "n": len(results)}

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results written to {args.output_dir}")
    os.unlink(tmp_config)


if __name__ == "__main__":
    main()

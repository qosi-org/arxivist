"""
FutureSim — Main Simulation Runner
====================================
Entry point for running FutureSim benchmark evaluations.

Paper reference: Section 4.1 (Experimental Setup), Appendix B.2 (Simulation Logic)

Usage:
  python run_simulation.py --config configs/config.yaml --agent native --model gpt-4o --seed 0
  python run_simulation.py --config configs/config.yaml --debug   # 10 questions, 5 days
  python run_simulation.py --config configs/config.yaml --dry-run # validate setup only
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent / "src"))

from futuresim.environment.sim_engine import SimulationEngine
from futuresim.corpus.retrieval import NewsRetriever
from futuresim.scoring.brier import compute_brier_skill_score, compute_accuracy
from futuresim.utils.config import load_config, set_seed
from futuresim.utils.logging import get_logger

logger = get_logger("futuresim.runner")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FutureSim: Replaying World Events to Evaluate Adaptive Agents"
    )
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument(
        "--agent",
        default="native",
        choices=["native", "custom", "external"],
        help="Harness type: native (paper default), custom (our baseline), external",
    )
    p.add_argument("--model", default="gpt-4o", help="Model name for API calls")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use 10 questions and 5 simulation days",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize all components and exit without running simulation",
    )
    p.add_argument("--resume", default=None, help="Path to checkpoint JSON to resume from")
    p.add_argument("--output-dir", default="results/", help="Directory for results output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    logger.info(f"FutureSim | model={args.model} | harness={args.agent} | seed={args.seed}")

    # --- Initialize retriever ---
    retriever = NewsRetriever(
        index_path=cfg.corpus.index_path,
        embedding_model_name=cfg.corpus.embedding_model,
        chunks_per_query=cfg.corpus.chunks_per_query,
        chunk_size=cfg.corpus.chunk_size,
    )
    if not args.dry_run:
        retriever.connect()
        logger.info("Retriever connected.")

    # --- Initialize simulation engine ---
    engine = SimulationEngine(cfg)

    if args.debug:
        # Override to short debug window
        cfg.simulation.start_date = str(date.today())
        cfg.simulation.end_date = str(date.today() + timedelta(days=5))
        logger.info("DEBUG MODE: 5-day simulation window")

    engine.load_questions()
    logger.info(f"Loaded questions. Starting simulation: {cfg.simulation.start_date} → {cfg.simulation.end_date}")

    if args.dry_run:
        print("Dry run complete. All components initialized successfully.")
        print(f"  Engine: {engine}")
        print(f"  Retriever: {retriever}")
        return

    # --- Resume from checkpoint ---
    if args.resume:
        checkpoint_path = Path(args.resume)
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        engine.current_date = date.fromisoformat(checkpoint["current_date"])
        logger.info(f"Resumed from checkpoint: {engine.current_date}")

    # --- Main simulation loop ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    day_count = 0
    while not engine.is_complete():
        # Set date cap on retriever for leakage prevention (Appendix B.3)
        retriever.set_date_cap(engine.current_date)

        logger.info(f"Day {day_count}: {engine.current_date}")

        # TODO: In a full implementation, launch the agent subprocess in sandbox here.
        # The agent interacts with the environment via MCP tool calls (Appendix B.2).
        # For now, this is a scaffold — replace with actual agent invocation.
        print(
            f"[{engine.current_date}] Simulation day {day_count}. "
            "Agent subprocess would be launched here."
        )

        # Advance to next day
        summary = engine.next_day()
        day_count += 1

        # Save daily checkpoint
        checkpoint = {
            "current_date": str(engine.current_date),
            "day": day_count,
            "resolved_today": summary["resolved_today"],
        }
        with open(output_dir / f"checkpoint_day_{day_count:03d}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    # --- Final metrics ---
    final_metrics = engine.get_final_metrics()
    metrics_path = output_dir / f"final_metrics_seed{args.seed}.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*50)
    print(f"FutureSim Complete | {day_count} days simulated")
    print(f"  Mean BSS:  {final_metrics['mean_brier_skill_score']:.4f}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Resolved:  {final_metrics['num_resolved']} / {final_metrics['num_questions']}")
    print(f"  Results:   {metrics_path}")
    print("="*50)


if __name__ == "__main__":
    main()

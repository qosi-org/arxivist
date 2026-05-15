"""
FutureSim — Question Generation Script
=========================================
Generates a forecasting question dataset from CCNews articles.

Paper reference: Appendix A.1 (Question Creation Methodology)
  Starting pool: 10,000+ Al Jazeera articles → 330 high-quality questions (~3% yield)

Usage:
  python generate_questions.py \\
      --articles-path data/ccnews/ \\
      --output-csv data/questions.csv \\
      --model gpt-4o \\
      --n-questions 500
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from futuresim.question_gen.generator import QuestionGenerator
from futuresim.utils.logging import get_logger

logger = get_logger("futuresim.generate_questions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate FutureSim forecasting questions from CCNews")
    p.add_argument("--articles-path", required=True, help="CCNews corpus root directory")
    p.add_argument("--output-csv", required=True, help="Output questions CSV path")
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM for question generation (paper uses frontier models)",
    )
    p.add_argument(
        "--n-questions",
        type=int,
        default=500,
        help="Target number of valid questions to generate",
    )
    p.add_argument(
        "--simulation-start",
        default="2026-01-01",
        help="Questions must resolve on or after this date",
    )
    p.add_argument(
        "--simulation-end",
        default="2026-03-28",
        help="Questions must resolve on or before this date",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set.")

    logger.info(
        f"Generating questions: articles={args.articles_path}, "
        f"target={args.n_questions}, window={args.simulation_start}→{args.simulation_end}"
    )

    gen = QuestionGenerator(
        model=args.model,
        api_key=api_key,
        simulation_start=args.simulation_start,
        simulation_end=args.simulation_end,
    )
    n = gen.generate_dataset(
        articles_path=args.articles_path,
        output_csv=args.output_csv,
        target_n=args.n_questions,
    )
    logger.info(f"Done: {n} questions written to {args.output_csv}")


if __name__ == "__main__":
    main()

"""
Multi-Agent Harness
====================
Prompt extensions for multi-agent FutureSim runs.

Paper reference: Section 5.5, Appendix E.3
  Three identical agents compete simultaneously; predictions averaged into a
  crowd aggregate shared with all agents each day.
  Agents are graded on time-weighted PEER score (relative to other agents).
"""

from __future__ import annotations

MULTI_AGENT_EXTENSION = """
## MULTI-AGENT SETTING
You are competing against {n_other} other forecasting agent(s) on the same set of questions.

You each predict independently on every wakeup day. After each day, your predictions
are averaged with the others' into a market aggregate (the `market_aggregate` column),
which you can see starting the following day.

You are scored relative to your competitors: to earn a positive time-weighted peer
score, your predictions need to be more accurate than the group average.

## SCORING (Time-Weighted Peer Score — Brier-Skill Based)
- **Time-Weighted Peer Score (TW-Peer)**: On each day a prediction is held, your
  Brier Skill Score is compared to the mean of all other agents' scores for the same
  question. These daily differences are summed over the lifetime of the prediction.
  A positive TW-Peer indicates predictions that were consistently more accurate than
  the group average.

**Relative Performance (multi-agent)**: Final scoring is relative, so you have to
outperform the market aggregate to gain positive peer score.

Note: `market_aggregate` and `my_prediction` columns contain Python dicts (or None).
  - `market_aggregate`: the mean probability distribution across all agents' latest
    predictions from the previous day. None on the first day.
  - `my_prediction`: your own latest forecast, or None if not yet predicted.
"""


def build_multi_agent_prompt(base_prompt: str, n_agents: int) -> str:
    """
    Append the multi-agent extension to a base harness prompt.

    Args:
        base_prompt: Native or custom harness prompt string
        n_agents: Total number of agents in this run

    Returns:
        Full prompt with multi-agent section inserted before "Begin."
    """
    extension = MULTI_AGENT_EXTENSION.format(n_other=n_agents - 1)
    if "Begin." in base_prompt:
        return base_prompt.replace("Begin.", extension + "\nBegin.")
    return base_prompt + "\n" + extension

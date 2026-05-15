"""
Brier Skill Score and Related Metrics
======================================
Implements the scoring rules from:
  "FutureSim: Replaying World Events to Evaluate Adaptive Agents"
  Section 3 (Evaluation) and Appendix C (Metrics)

Equations implemented:
  - BSS(q) = 1 - Σ_{o ∈ Ω_q ∪ {y_q}} (p_q(o) - 1[o = y_q])²   [Section 3]
  - Top-1 Accuracy                                                 [Section 3]
  - Time-Weighted Score (TW)                                       [Appendix C.3]
  - Time-Weighted Peer Score (TWPeer)                              [Appendix C.3]
  - Aggregate Prediction (multi-agent)                             [Appendix C.4]
  - Total Variation Distance                                       [Appendix C.4]
"""

from __future__ import annotations

from typing import Callable


def compute_brier_skill_score(
    prediction: dict[str, float],
    ground_truth: str,
    match_fn: Callable[[str, str], bool] | None = None,
) -> float:
    """
    Compute the Brier Skill Score for a single question.

    Eq. from Section 3:
      BSS(q) = 1 - Σ_{o ∈ Ω_q ∪ {y_q}} (p_q(o) - 1[o = y_q])²

    This is a proper scoring rule (Theorem C.1):
    - 1.0  = fully confident and correct
    - 0.0  = abstaining (no probability mass submitted)
    - -1.0 = all probability mass on wrong outcomes

    Args:
        prediction: dict mapping outcome strings to probabilities (sum ≤ 1.0).
                    Empty dict means the agent abstained → score = 0.0
        ground_truth: the resolved answer string
        match_fn: optional callable(predicted_outcome, ground_truth) → bool
                  for semantic matching. Defaults to exact string equality.

    Returns:
        BSS value in [-1.0, 1.0]
    """
    if match_fn is None:
        match_fn = lambda a, b: a.strip().lower() == b.strip().lower()

    # Ω_q ∪ {y_q}: all outcomes the agent named, plus the ground truth
    all_outcomes: set[str] = set(prediction.keys()) | {ground_truth}

    squared_error_sum = 0.0
    for outcome in all_outcomes:
        p = prediction.get(outcome, 0.0)
        y = 1.0 if match_fn(outcome, ground_truth) else 0.0
        squared_error_sum += (p - y) ** 2

    return 1.0 - squared_error_sum


def compute_accuracy(
    prediction: dict[str, float],
    ground_truth: str,
    match_fn: Callable[[str, str], bool] | None = None,
) -> float:
    """
    Top-1 Accuracy: 1 if the highest-probability outcome matches ground truth.

    Eq. from Section 3:
      Acc = (1/|Q|) Σ_{q∈Q} 1[argmax_o p_q(o) = y_q]

    Args:
        prediction: dict mapping outcome → probability
        ground_truth: resolved answer
        match_fn: optional semantic match function

    Returns:
        1.0 if correct, 0.0 if incorrect or no prediction.
    """
    if not prediction:
        return 0.0
    if match_fn is None:
        match_fn = lambda a, b: a.strip().lower() == b.strip().lower()

    top_outcome = max(prediction, key=prediction.get)
    return 1.0 if match_fn(top_outcome, ground_truth) else 0.0


def compute_time_weighted_score(
    daily_bss: dict[str, list[float]],
    question_open_days: dict[str, int],
) -> float:
    """
    Time-Weighted Score across all questions.

    Eq. from Appendix C.3:
      TW = 100 * Σ_{q∈Q} (1/|T_q|) * Σ_{t∈T_q} BSS_t(q)

    Args:
        daily_bss: {qid: [BSS on each open day]}  — 0.0 if no forecast held that day
        question_open_days: {qid: total days question was open}

    Returns:
        Time-weighted score (× 100 as per paper convention)
    """
    total = 0.0
    for qid, daily_scores in daily_bss.items():
        n_days = question_open_days.get(qid, len(daily_scores))
        if n_days == 0:
            continue
        total += sum(daily_scores) / n_days
    return 100.0 * total


def compute_peer_score(
    agent_daily_bss: list[float],
    others_daily_bss: list[list[float]],
    question_open_days: int,
) -> float:
    """
    Time-Weighted Peer Score for a single question (multi-agent setting).

    Eq. from Appendix C.3:
      Peer_{a,t}(q) = BSS_{a,t}(q) - BSS_{-a,t}(q)
      TWPeer_a = 100 * Σ_q (1/|T_q|) * Σ_t Peer_{a,t}(q)

    Args:
        agent_daily_bss: this agent's daily BSS for one question
        others_daily_bss: list of other agents' daily BSS lists for same question
        question_open_days: |T_q|

    Returns:
        Per-question peer contribution (before × 100 scaling)
    """
    assert len(agent_daily_bss) == question_open_days
    daily_peer = []
    for t in range(question_open_days):
        others_at_t = [scores[t] for scores in others_daily_bss if t < len(scores)]
        baseline = sum(others_at_t) / len(others_at_t) if others_at_t else 0.0
        daily_peer.append(agent_daily_bss[t] - baseline)
    return sum(daily_peer) / question_open_days


def compute_aggregate_prediction(
    agent_predictions: list[dict[str, float]],
) -> dict[str, float]:
    """
    Coordinate-wise mean of agent predictions (multi-agent crowd aggregate).

    Eq. from Appendix C.4:
      p̄_q(o) = (1/n_q) * Σ_{a=1}^{n_q} p_q^(a)(o)
      Missing outcomes treated as probability 0.

    Args:
        agent_predictions: list of prediction dicts from each agent

    Returns:
        Aggregate prediction dict
    """
    all_outcomes: set[str] = set()
    for pred in agent_predictions:
        all_outcomes.update(pred.keys())

    n = len(agent_predictions)
    aggregate: dict[str, float] = {}
    for outcome in all_outcomes:
        aggregate[outcome] = sum(pred.get(outcome, 0.0) for pred in agent_predictions) / n
    return aggregate


def compute_total_variation_distance(
    pred_a: dict[str, float],
    pred_b: dict[str, float],
) -> float:
    """
    Total Variation Distance between two forecasts.

    Eq. from Appendix C.4:
      d_TV(p_q, p'_q) = (1/2) * Σ_{o ∈ Ω_q ∪ Ω'_q} |p_q(o) - p'_q(o)|

    Args:
        pred_a, pred_b: two prediction dicts

    Returns:
        TV distance in [0, 1]
    """
    all_outcomes = set(pred_a.keys()) | set(pred_b.keys())
    return 0.5 * sum(
        abs(pred_a.get(o, 0.0) - pred_b.get(o, 0.0)) for o in all_outcomes
    )

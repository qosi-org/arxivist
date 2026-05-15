"""
Agent Base Classes and Harness Templates
==========================================
Defines the abstract agent interface and harness configurations for FutureSim.

Paper reference: Section 3.1 (Agent Interaction), Appendix B.1 (Our Baseline Harness),
                 Appendix E.1 (Native Harness), Appendix E.2 (Our Harness)

An "agent" = the tools, prompts, and orchestration a model uses to interact with the environment.
The environment provides only two actions: submit_forecast() and next_day().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from typing import Optional


class BaseAgent(ABC):
    """
    Abstract interface for a FutureSim forecasting agent.

    Subclasses implement interact() which is called once per simulation day.
    The agent must eventually call env.submit_forecast() and env.next_day().

    Paper reference: Section 3.1 (Agent Interaction)
    """

    def __init__(self, model_name: str, workspace: str):
        """
        Args:
            model_name: Identifier of the underlying LLM
            workspace: Path to agent's sandboxed workspace directory
        """
        self.model_name = model_name
        self.workspace = Path(workspace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

    @abstractmethod
    def interact(
        self,
        current_date: str,
        next_date: str,
        num_active: int,
        num_resolved: int,
        new_articles_count: int,
        resolved_feedback: list[dict],
    ) -> None:
        """
        Called once per simulation day. The agent should:
        1. Read market.csv to identify questions needing updates
        2. Search news corpus for relevant evidence
        3. Submit/update forecasts via submit_forecast()
        4. Call next_day() to advance simulation
        """
        ...


# --- Native Harness Prompt Template (Appendix E.1) ---

NATIVE_HARNESS_PROMPT = Template("""You are a forecasting agent. Today is $current_date. \
Your goal is to make accurate and calibrated predictions.

## UPDATE CADENCE
You have the chance to update your predictions every $timegap_days day(s). \
Your workspace files (memory/, scripts, notes) persist across days -- use them to \
track reasoning and lessons learned. Articles are available via the search tool and \
in the articles/ directory. Current date: $current_date. Next scheduled update: $next_date.

## SCORING (Brier Skill Score)
You have to output a distribution of (outcome, probability) pairs for each question you make a forecast on.
You are evaluated on the Brier Skill Score = 1 - sum_i (p_i - y_i)^2 summed over all outcomes, where:
- p_i = your probability for outcome i
- y_i = 1 if your outcome i is TRUE, 0 otherwise
- Higher is better: 1.0 = perfect, 0.0 = abstaining from guessing, negative = worse than abstaining.

Key Mechanics:
1. Accuracy + Calibration: assign calibrated probabilities that reflect true likelihoods.
2. Time-Weighted Score: forecasts made earlier matter, but updating is rewarded when new evidence arrives.
3. Prediction-Count Incentive: unanswered active questions receive zero contribution.
4. End-of-Session Metrics are shown after each session.
5. Max Outcomes: submit at most $max_outcomes_per_question outcomes per question.
6. No Placeholders: "Unknown", "TBD", and "Other" hurt your score.

## AVAILABLE DATA
You have access to a news article database, which is updated daily through a search tool, \
that you can use to find evidence for your forecasts.
You can access the market.csv file (READ-ONLY) in your workspace containing \
$num_questions questions ($num_active active/unresolved, $num_resolved resolved).

## TOOLS AVAILABLE FOR YOUR USE
- mcp__forecast__search_news(query, from_date?, to_date?): search the news corpus for evidence.
- mcp__forecast__submit_forecasts(question_id, outcomes): submit exactly one forecast for one question ID.
- mcp__forecast__next_day(): end the current session and proceed to the next one.

## SUBMISSION RULES
- qid must be from an active unresolved question identified from market.csv.
- Maximum of $max_outcomes_per_question outcomes allowed per question.
- Outcome names must be real predicted answers.
- Never use placeholders like "Unknown", "TBD", "Other", or "N/A".
- Probabilities must sum to <= 1.0.

Begin.""")


# --- Custom Baseline Harness Prompt (Appendix E.2, simplified) ---

CUSTOM_HARNESS_PROMPT = Template("""You are a forecasting agent. Today is $current_date. \
Your goal is to make accurate and calibrated predictions.

$results_block

## UPDATE CADENCE
You can make updates every $timegap_days day(s). Your context is cleared after every session \
and your memory (along with past predictions) is the only information retained between sessions.
$new_articles_text
Current date: $current_date. Next scheduled update: $next_date.
$resolution_reminder

IMPORTANT: You have predictions on $predicted_count out of $active_count active questions.

UPDATE RULES:
- Do NOT re-predict questions from scratch unless you find specific new evidence.
- Only update a prediction if you find SPECIFIC NEW evidence (news, data) that updates your view.

PRIORITIES FOR UPDATES:
1. Questions resolving the next day -- make sure your prediction is up-to-date before calling next_day.
2. Questions without predictions (if any)
3. Questions where today's news search reveals new information
4. Questions approaching resolution date that you haven't checked recently
5. Skip questions where there is no new evidence

## YOUR MEMORY
Current meta-insights with their indices:
$meta_insight_index

## SCORING (Brier Skill Score)
BSS = 1 - Σ (p_i - y_i)² | Higher=better: 1.0=perfect, 0.0=abstain, negative=worse than abstain.
Max $max_outcomes_per_question outcomes per question. Probabilities must sum to ≤ 1.0.
No Placeholders: "Unknown", "TBD", "Other" hurt your score.

## TOOLS
- mcp__forecast__search_news(query, from_date?, to_date?)
- mcp__forecast__memory_retrieve / _new / _update / _delete  (meta-insights)
- mcp__forecast__mem_add / _update / _delete  (per-question notes)
- mcp__forecast__submit_forecasts(question_id, outcomes)
- mcp__forecast__next_day()  [first call enters memory-update mode; second call advances day]

Budget at start:
Actions remaining: $max_actions
Context tokens remaining: $max_total_tokens

Begin.""")


def build_native_prompt(
    current_date: str,
    next_date: str,
    timegap_days: int,
    num_questions: int,
    num_active: int,
    num_resolved: int,
    max_outcomes_per_question: int = 5,
) -> str:
    """Render the native harness prompt (Appendix E.1)."""
    return NATIVE_HARNESS_PROMPT.substitute(
        current_date=current_date,
        next_date=next_date,
        timegap_days=timegap_days,
        num_questions=num_questions,
        num_active=num_active,
        num_resolved=num_resolved,
        max_outcomes_per_question=max_outcomes_per_question,
    )


def build_custom_prompt(
    current_date: str,
    next_date: str,
    timegap_days: int,
    num_questions: int,
    active_count: int,
    predicted_count: int,
    new_articles_text: str = "",
    results_block: str = "",
    resolution_reminder: str = "",
    meta_insight_index: str = "(none yet)",
    max_outcomes_per_question: int = 5,
    max_actions: int = 200,
    max_total_tokens: int = 200_000,
) -> str:
    """Render the custom baseline harness prompt (Appendix E.2)."""
    return CUSTOM_HARNESS_PROMPT.substitute(
        current_date=current_date,
        next_date=next_date,
        timegap_days=timegap_days,
        num_questions=num_questions,
        active_count=active_count,
        predicted_count=predicted_count,
        new_articles_text=new_articles_text,
        results_block=results_block,
        resolution_reminder=resolution_reminder,
        meta_insight_index=meta_insight_index,
        max_outcomes_per_question=max_outcomes_per_question,
        max_actions=max_actions,
        max_total_tokens=max_total_tokens,
    )

"""
evolvemem/memory/extractor.py

Sliding-window LLM-based memory extractor with extraction quality guards.
Implements Section 3.1 "Memory extraction" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Three quality guards (Appendix A, Equation 6):
  1. Retry on LLM failure with increasing wait intervals
  2. Chunk-split into sub-windows when context limit exceeded
  3. Coverage verifier triggers re-extraction for missing keywords
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

from .store import MemoryUnit

logger = logging.getLogger(__name__)


_EXTRACTION_SYSTEM_PROMPT = """You are a professional information extraction assistant.
Extract ALL valuable information from the following dialogue into structured memory entries."""

_EXTRACTION_USER_TEMPLATE = """
{context}

[Dialogue from {date}]
{dialogue_text}

[Requirements]
1. Complete Coverage: Generate entries for ALL facts, events, opinions, plans, feelings.
2. Force Disambiguation: PROHIBIT pronouns (he/she/it/they). Use actual names and absolute dates.
3. Lossless Restatement: Each entry must be complete, independent, self-contained.
4. Extract EVERY specific detail – no paraphrasing fine-grained facts:
   - Named entities: book/movie/song/game titles, brand names, places, pet names, nicknames
   - Quantities: exact counts, frequencies ("twice"), durations ("for 3 years", "since 2019")
   - Lists: if someone mentions multiple items, create ONE entry that lists them ALL
   - Gifts and possessions: who gave what to whom, when
5. Cover names, places, objects, opinions, plans, feelings, events, dates, gifts, hobbies,
   relationships, pets, travel, food, books, art, music, work, family, health.

[Output Format]
Return a JSON array:
[
  {{
    "lossless_restatement": "Complete sentence with all subjects, objects, time, location",
    "keywords": ["keyword1", "keyword2"],
    "timestamp": "YYYY-MM-DD or null",
    "location": "location or null",
    "persons": ["name1", "name2"],
    "entities": ["entity1"],
    "topic": "topic phrase",
    "memory_type": "episodic|semantic|preference|project_state|working_summary|procedural"
  }}
]
Return ONLY the JSON array. Extract at least 15 entries. Prioritise completeness over brevity.
"""

_TARGETED_EXTRACTION_TEMPLATE = """
Re-examine this dialogue specifically for information about: {missing_topics}

[Dialogue]
{dialogue_text}

Return a JSON array of memory entries covering ONLY the missing topics above.
Use the same format as before. Return ONLY JSON.
"""


class MemoryExtractor:
    """
    Sliding-window LLM-based memory extractor.

    Implements the extraction pipeline from Section 3.1:
      - Overlapping W-turn windows over source conversation S
      - Three quality guards: retry, chunk-split, coverage verify
      - Targeted re-extraction for coverage gaps (Equation 6 / 15)

    Paper reference: Section 3.1 "Memory extraction", Appendix A Equation 6
    """

    def __init__(
        self,
        llm_client,
        window_size: int = 40,
        sub_window_size: int = 15,
        max_retries: int = 3,  # ASSUMED: paper says "increasing wait intervals" without count
    ):
        """
        Args:
            llm_client: LLMClient instance for extraction calls.
            window_size: W — turns per sliding window (Section 3.1: W=40).
            sub_window_size: C — fallback sub-window size when context exceeded (Appendix A: C=15).
            max_retries: Retry limit for failed LLM calls (ASSUMED=3).
        """
        self.llm_client = llm_client
        self.window_size = window_size
        self.sub_window_size = sub_window_size
        self.max_retries = max_retries

    def extract(
        self,
        conversation: List[dict],
        existing_memories: Optional[List[MemoryUnit]] = None,
        session_date: str = "unknown date",
    ) -> List[MemoryUnit]:
        """
        Extract memory units from a conversation via sliding window.

        Args:
            conversation: List of turn dicts with 'speaker' and 'text' keys.
            existing_memories: Already-extracted units (for dedup context).
            session_date: ISO date string for this session.

        Returns:
            List of extracted MemoryUnit objects.

        Paper reference: Section 3.1, Appendix A Equation 6
        """
        all_units: List[MemoryUnit] = []

        # Build overlap context from existing memories for dedup
        existing_context = ""
        if existing_memories:
            sample = existing_memories[-5:]  # last 5 as context
            existing_context = "Previous extractions (avoid duplicating):\n" + \
                "\n".join(f"- {m.content}" for m in sample)

        # Sliding window over conversation turns
        for start in range(0, len(conversation), self.window_size // 2):  # 50% overlap
            window = conversation[start: start + self.window_size]
            if not window:
                break

            dialogue_text = self._format_turns(window)

            # Guard 1 + 2: retry + chunk-split
            units = self._extract_window_with_guards(
                dialogue_text=dialogue_text,
                context=existing_context,
                date=session_date,
            )

            # Guard 3: coverage verification
            missing_keywords = self._coverage_verify(units, dialogue_text)
            if missing_keywords:
                logger.info(f"Coverage gap detected: {missing_keywords}. Triggering re-extraction.")
                extra = self._targeted_extract(dialogue_text, missing_keywords)
                units.extend(extra)

            all_units.extend(units)
            existing_context = ""  # Only use for first window

        return all_units

    def extract_targeted(
        self,
        conversation: List[dict],
        missing_keywords: List[str],
    ) -> List[MemoryUnit]:
        """
        Targeted re-extraction for specific missing keywords.

        Called by the evolution engine when the diagnosis detects coverage gaps.
        Implements K_{r+1} = K_r ∪ φ^targeted_ext(S, V^miss_r) from Appendix A Equation 15.

        Args:
            conversation: Source conversation.
            missing_keywords: Keywords that should be covered but aren't.

        Returns:
            List of newly extracted MemoryUnit objects.
        """
        full_text = self._format_turns(conversation)
        return self._targeted_extract(full_text, missing_keywords)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_window_with_guards(
        self,
        dialogue_text: str,
        context: str,
        date: str,
    ) -> List[MemoryUnit]:
        """
        Extract from a single window with retry (Guard 1) and chunk-split (Guard 2).

        Implements Equation 6 branches:
          - Primary: φ_ext(S^(j)) with retries
          - Fallback: split into C-turn sub-windows and merge
        """
        prompt = _EXTRACTION_USER_TEMPLATE.format(
            context=context,
            date=date,
            dialogue_text=dialogue_text,
        )

        # Guard 1: retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                raw = self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system=_EXTRACTION_SYSTEM_PROMPT,
                )
                return self._parse_extraction_response(raw)
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                time.sleep(wait)

        # Guard 2: chunk-split fallback
        logger.warning("All retries exhausted. Falling back to sub-window extraction.")
        return self._extract_subwindows(dialogue_text, context, date)

    def _extract_subwindows(
        self,
        dialogue_text: str,
        context: str,
        date: str,
    ) -> List[MemoryUnit]:
        """Guard 2: split dialogue into C-turn sub-windows and merge results."""
        lines = dialogue_text.split("\n")
        all_units = []
        chunk_size = self.sub_window_size

        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i: i + chunk_size])
            if not chunk.strip():
                continue
            prompt = _EXTRACTION_USER_TEMPLATE.format(
                context=context,
                date=date,
                dialogue_text=chunk,
            )
            try:
                raw = self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system=_EXTRACTION_SYSTEM_PROMPT,
                )
                all_units.extend(self._parse_extraction_response(raw))
            except Exception as e:
                logger.error(f"Sub-window extraction failed: {e}")

        return all_units

    def _targeted_extract(
        self,
        dialogue_text: str,
        missing_keywords: List[str],
    ) -> List[MemoryUnit]:
        """Guard 3: targeted re-extraction for missing keywords."""
        prompt = _TARGETED_EXTRACTION_TEMPLATE.format(
            missing_topics=", ".join(missing_keywords),
            dialogue_text=dialogue_text,
        )
        try:
            raw = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_EXTRACTION_SYSTEM_PROMPT,
            )
            return self._parse_extraction_response(raw)
        except Exception as e:
            logger.error(f"Targeted extraction failed: {e}")
            return []

    def _coverage_verify(
        self,
        units: List[MemoryUnit],
        source_text: str,
    ) -> List[str]:
        """
        Guard 3: compare extracted keywords against source text.

        The verifier compares extracted memories against reference keywords from the source
        text and returns the missing subset V^miss, which triggers targeted re-extraction.

        Paper reference: Section 3.1, Appendix A Equation 6 (third branch)
        """
        # Extract capitalized named entities from source as reference keywords
        reference_keywords = set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b", source_text))
        if len(reference_keywords) < 3:
            return []  # Too few reference keywords to verify

        # Collect all keywords from extracted units
        extracted_keywords = set()
        for unit in units:
            extracted_keywords.update(unit.keywords)
            extracted_keywords.update(unit.entities)
            extracted_keywords.update(unit.persons)

        missing = reference_keywords - extracted_keywords
        # Filter to only "important-looking" missing keywords (len > 3)
        missing = [k for k in missing if len(k) > 3]
        return missing[:10]  # Cap at 10 to avoid over-triggering

    def _parse_extraction_response(self, raw: str) -> List[MemoryUnit]:
        """Parse LLM JSON response into MemoryUnit objects."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()

        try:
            entries = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction JSON. Attempting repair.")
            # Try to extract array from partial response
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    entries = json.loads(match.group())
                except Exception:
                    return []
            else:
                return []

        units = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = entry.get("lossless_restatement", "").strip()
            if not content or len(content) < 3:
                continue

            units.append(MemoryUnit(
                content=content,
                memory_type=entry.get("memory_type", "episodic"),
                keywords=entry.get("keywords", []),
                timestamp=entry.get("timestamp"),
                locations=[entry["location"]] if entry.get("location") else [],
                persons=entry.get("persons", []),
                entities=entry.get("entities", []),
                topics=[entry["topic"]] if entry.get("topic") else [],
            ))

        return units

    @staticmethod
    def _format_turns(turns: List[dict]) -> str:
        """Format conversation turns into a dialogue string."""
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", turn.get("content", ""))
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

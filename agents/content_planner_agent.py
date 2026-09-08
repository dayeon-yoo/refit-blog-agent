from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.editorial_context import build_safe_editorial_context
from agents.idea_agent import IdeaGenerator
from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea, ContentPlan


class ContentPlannerAgent:
    """Prepare a source-grounded, free-form script before article writing."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "content_planner.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    @classmethod
    def is_experience_like(cls, source: str, idea: BlogIdea) -> bool:
        """Route explicit experience pieces around the generative planner."""
        source_families = IdeaGenerator._source_format_families(source)
        format_families = IdeaGenerator._format_families(idea.content_format or "")
        return "experience" in source_families or "experience" in format_families

    @classmethod
    def planning_mode(cls, source: str, idea: BlogIdea) -> str:
        """Select planning authority without an extra classification call."""
        source_families = IdeaGenerator._source_format_families(source)
        if "experience" in source_families:
            return "experience_deterministic"

        # Raw source has priority over refinement metadata. Explicit guide,
        # information, and comparison requests retain the generative planner
        # even when styling language is also present in the source.
        structured_families = {"guide", "checklist", "informational", "comparison"}
        if source_families.intersection(structured_families):
            return "generative_structured"
        if source_families.intersection({"trend", "editorial", "styling"}):
            return "source_grounded_editorial"

        # Only ambiguous raw input falls back to the refined format.
        format_families = IdeaGenerator._format_families(idea.content_format or "")
        if "experience" in format_families:
            return "experience_deterministic"
        if format_families.intersection({"trend", "editorial", "styling"}):
            return "source_grounded_editorial"
        return "generative_structured"

    @staticmethod
    def _deterministic_experience_plan(source: str) -> ContentPlan:
        return ContentPlan(
            writing_script=(
                "[실제 경험과 사실]\n"
                "원본 메모에 명시된 경험과 행동만 사실로 사용한다."
                " 원본에 없는 구매 상황, 선택 이유, 감정, 반응, 장소의 분위기,"
                " 실제 착장 세부는 추가하지 않는다.\n\n"
                f"[원본 메모]\n{source}\n\n"
                "[일반적인 제안]\n"
                "원본에 없는 코디나 활용 아이디어가 필요하면 작성자의 경험이 아닌"
                " 독자를 위한 일반적인 제안 또는 조건부 설명으로만 작성한다.\n\n"
                "[전개]\n"
                "원본에 적힌 소재와 행동을 자연스러운 순서로 서술하고,"
                " 근거 없는 개인 에피소드로 내용을 확장하지 않는다."
            )
        )

    def plan(self, source: str, idea: BlogIdea) -> ContentPlan:
        source = (source or "").strip()
        if not source:
            raise ValueError("source is required for content planning")
        mode = self.planning_mode(source, idea)
        if mode == "experience_deterministic":
            return self._deterministic_experience_plan(source)
        editorial_context = build_safe_editorial_context(idea, source)
        payload = {
            "raw_source": source,
            "planning_mode": mode,
            "editorial_context": editorial_context,
            # Keep the legacy key for MockLLMClient compatibility, but never
            # pass the full BlogIdea through it.
            "refined_blog_idea": editorial_context,
        }
        prompt = self.prompt
        if mode == "source_grounded_editorial":
            prompt += (
                "\n\nPLANNING MODE: SOURCE-GROUNDED TREND/EDITORIAL. "
                "Reorder only the claims, questions, actions, and relationships "
                "already present in raw_source. Preserve each subject/object "
                "relationship and do not reinterpret one claim as another. Do not "
                "add shopping tips, checklists, external trend facts, environmental "
                "or economic effects, product examples, or personal episodes. "
                "Return a short flow of source content units, not a richer article plan."
            )
        result = self.client.generate_structured(prompt, ContentPlan, payload)
        if not result.writing_script.strip():
            raise ValueError("Content Planner returned an empty writing script")
        return result

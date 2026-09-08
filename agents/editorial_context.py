from __future__ import annotations

from models.schemas import BlogIdea


def build_safe_editorial_context(idea: BlogIdea, raw_source: str = "") -> dict[str, object]:
    """Keep editorial guidance separate from raw factual material."""
    context: dict[str, object] = {
        "title": idea.title,
        "keyword": idea.keyword,
        "content_format": idea.content_format,
        "target_reader": idea.target_reader,
    }
    source_text = f"{raw_source} {idea.title} {idea.keyword} {idea.rifit_connection}".lower()
    if idea.rifit_connection and any(
        term in source_text for term in ("재사용", "재활용", "순환", "기부", "수거", "빈티지", "헌옷")
    ):
        context["rifit_editorial_direction"] = (
            "의류 재사용과 순환이라는 맥락에서 주제와 자연스럽게 연결되는 경우 "
            "리핏을 언급하되, "
            "제공되지 않은 서비스 기능이나 외부 사실은 설명하지 않는다."
        )
    return context

from __future__ import annotations

from agents.writer_agent import WriterAgent
from models.schemas import BlogIdea, ScoredIdea


def make_idea(title: str, keyword: str, angle: str) -> ScoredIdea:
    idea = BlogIdea(
        title=title,
        keyword=keyword,
        search_intent="정보 탐색",
        angle=angle,
        rifit_connection="",
        seasonality=0.5,
    )
    scored = ScoredIdea(
        idea_id="test-1",
        scores={"s": 1},
        total_score=50,
        reason="mock",
        idea=idea,
    )
    return scored


def test_writer_generates_long_structured_post():
    writer = WriterAgent()
    scored = make_idea("헌옷 보낼 때 이것만은 꼭 확인하세요", "헌옷 처리", "헌옷 처리 - 기부·수거·정리 단계에서의 실용 가이드")
    post = writer.write(scored)

    # title preserved
    assert post.title == scored.idea.title

    # content length sufficiently larger than trivial
    assert len(post.content) > 400

    # at least 4 subheadings
    headings = [l for l in post.content.splitlines() if l.startswith('## ')]
    assert len(headings) >= 4

    # forbidden internal phrases not present
    forbidden = ["검색 의도", "seo", "prompt", "agent", "brand fit", "브랜드 평가", "critic", "ai 평가", "콘텐츠 생성 과정", "검색 유입을 위해"]
    low = post.content.lower()
    for f in forbidden:
        assert f not in low


def test_writer_does_not_use_fixed_template_headings():
    writer = WriterAgent()
    scored = make_idea("옷장 정리 팁", "옷장 정리", "옷장 정리 - 공간 구성과 분류 기준, 정리 루틴 중심의 실용적 접근")
    post = writer.write(scored)

    # Ensure disallowed fixed headings are not used verbatim
    disallowed = ["왜 이 주제가 중요한가요?", "실용 팁", "오늘 바로 해볼 수 있는 행동"]
    for d in disallowed:
        assert d not in post.content

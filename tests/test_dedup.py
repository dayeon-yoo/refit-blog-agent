from pathlib import Path
import json

from models.schemas import BlogIdea
from orchestrator.dedupe import is_duplicate, filter_duplicates, load_past_items


def make_idea(title: str, keyword: str, angle: str) -> BlogIdea:
    return BlogIdea(title=title, keyword=keyword, search_intent="info", angle=angle, rifit_connection="x", seasonality=0.5)


def test_no_past_content_not_duplicate(tmp_path, monkeypatch):
    # point outputs to empty temp dir
    monkeypatch.chdir(tmp_path)
    past = load_past_items(Path("outputs"))
    assert past["ideas"] == []
    idea = make_idea("Unique Title A", "kw-a", "angle a")
    assert not is_duplicate(idea, past["ideas"], past["posts"])


def test_identical_title_is_duplicate():
    past = {"ideas": [{"title": "Same Title", "keyword": "k", "angle": "a"}], "posts": []}
    idea = make_idea("Same Title", "k2", "b")
    assert is_duplicate(idea, past["ideas"], past["posts"]) is True


def test_same_keyword_different_angle_not_duplicate():
    past = {"ideas": [{"title": "Old Title", "keyword": "same-kw", "angle": "angle-one"}], "posts": []}
    idea = make_idea("New Title", "same-kw", "completely different angle")
    assert is_duplicate(idea, past["ideas"], past["posts"]) is False


def test_same_keyword_similar_angle_is_duplicate():
    past = {"ideas": [{"title": "Old Title", "keyword": "same-kw", "angle": "실용적인 옷장 정리/관리와 적용 방법"}], "posts": []}
    idea = make_idea("New Title", "same-kw", "실용적인 옷장 관리와 적용 방법을 소개합니다")
    assert is_duplicate(idea, past["ideas"], past["posts"]) is True


def test_filter_duplicates_integration(tmp_path):
    # create an outputs/history file containing a past idea that will match generated candidates
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    hist = outputs / "history"
    hist.mkdir()
    payload = {
        "ideas": [
            {
                "title": "의류 재사용과 리셀 팁 전 체크할 것",
                "keyword": "의류 재사용 방법",
                "search_intent": "문제 해결",
                "angle": "의류 재사용과 리셀 팁에 대한 현실적인 문제 해결과 의류 순환 관점의 정리",
                "rifit_connection": "x",
                "seasonality": 0.9,
            }
        ],
        "blog_posts": [],
    }
    f = hist / "hist1.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # prepare candidate that should be filtered
    candidates = [
        make_idea("의류 재사용과 리셀 팁 전 체크할 것", "의류 재사용 방법", "의류 재사용과 리셀 팁에 대한 현실적인 문제 해결과 의류 순환 관점의 정리"),
        make_idea("완전히 다른 아이디어", "other-kw", "다른 각도 설명"),
    ]

    filtered = filter_duplicates(candidates, outputs_dir=outputs)
    titles = [c.title for c in filtered]
    assert "의류 재사용과 리셀 팁 전 체크할 것" not in titles
    assert "완전히 다른 아이디어" in titles

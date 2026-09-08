from agents.qc_agent import QCAgent
from llm.client import MockLLMClient
from models.schemas import BlogIdea, BlogPost, ImagePlan, ImagePrompt


def make_idea() -> BlogIdea:
    return BlogIdea(
        title="빈티지 체크셔츠 출근 코디",
        keyword="빈티지 체크셔츠",
        search_intent="정보 탐색",
        angle="빈티지 체크셔츠를 출근룩으로 활용하는 경험",
        rifit_connection="입지 않는 옷을 다시 활용하는 흐름에서 의류 순환을 소개",
        seasonality=0.5,
        content_format="경험 후기",
        content_perspective="개인 경험",
    )


def make_post(content: str, tags=None, image_text="빈티지 체크셔츠를 활용한 출근 스타일링") -> BlogPost:
    return BlogPost(
        title="빈티지 체크셔츠 출근 코디",
        keyword="빈티지 체크셔츠",
        content=content,
        summary="빈티지 체크셔츠를 출근 코디로 활용한 이야기입니다.",
        cta="가지고 있는 옷을 다시 활용해보세요.",
        tags=tags if tags is not None else ["빈티지체크셔츠", "체크셔츠코디", "인턴출근룩"],
        image_plan=ImagePlan(images=[ImagePrompt(
            placement="hero-intro",
            purpose=image_text,
            prompt=image_text,
            image_type="styling",
            alt_text=image_text,
        )]),
    )


def test_clean_experience_passes_without_blocking_issue():
    source = "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다."
    result = QCAgent(MockLLMClient()).check(source, make_idea(), make_post("오슬로에서 산 체크셔츠로 출근 코디를 해봤습니다. 체크셔츠는 슬랙스와 매치해볼 수 있습니다."))

    assert result.status == "PASS"
    assert result.issues == []


def test_fabricated_personal_reaction_is_blocked():
    source = "빈티지 체크셔츠로 출근 코디를 해봤다."
    result = QCAgent(MockLLMClient()).check(source, make_idea(), make_post("회사 동료들이 모두 셔츠가 예쁘다고 칭찬해줬어요."))

    assert result.status == "BLOCK"
    assert any(issue.category == "source_grounding" for issue in result.issues)


def test_internal_metadata_is_blocked():
    post = make_post("검색 의도는 정보 탐색이며 SEO 전략은 본문에 반영했습니다.")
    result = QCAgent(MockLLMClient()).check("빈티지 체크셔츠 출근룩", make_idea(), post)

    assert result.status == "BLOCK"
    issue = next(issue for issue in result.issues if issue.category == "metadata_leakage")
    assert "검색 의도" in issue.evidence
    assert "SEO" in issue.evidence


def test_rifit_issue_points_to_the_offending_sentence():
    content = (
        "빈티지 체크셔츠를 활용해 출근 코디를 해봤습니다.\n\n"
        "빈티지 제품을 재사용하려는 마음이 있다면, 리핏 같은 구매 서비스를 알아보는 것도 좋겠어요."
    )
    result = QCAgent(MockLLMClient()).check(
        "빈티지 체크셔츠로 출근 코디를 해봤다.", make_idea(), make_post(content)
    )

    issue = next(issue for issue in result.issues if issue.category == "rifit_grounding")
    assert issue.evidence == "빈티지 제품을 재사용하려는 마음이 있다면, 리핏 같은 구매 서비스를 알아보는 것도 좋겠어요."
    assert "빈티지 체크셔츠를 활용해 출근 코디를 해봤습니다." not in issue.evidence


def test_malformed_title_and_repeated_sentence_need_revision():
    post = make_post("같은 문장입니다. 같은 문장입니다.")
    post.title = "인턴 출근룩으로를 고를 때 확인한 기준"
    result = QCAgent(MockLLMClient()).check("인턴 출근룩", make_idea(), post)

    assert result.status == "NEEDS_REVISION"
    assert {issue.category for issue in result.issues} >= {"writing_quality"}


def test_duplicate_issue_points_to_the_repeated_sentence():
    repeated = "같은 장면을 반복해서 사용했습니다."
    result = QCAgent(MockLLMClient()).check(
        "빈티지 체크셔츠", make_idea(), make_post(f"{repeated}\n{repeated}")
    )

    issue = next(issue for issue in result.issues if issue.category == "writing_quality")
    assert issue.evidence == repeated


def test_unrelated_tags_are_reported_without_reimplementing_ranking():
    post = make_post("빈티지 체크셔츠로 출근 코디를 소개합니다.", tags=["가능한패션", "뒤재사용"])
    result = QCAgent(MockLLMClient()).check("빈티지 체크셔츠 출근룩", make_idea(), post)

    assert result.status == "NEEDS_REVISION"
    assert any(issue.category == "tag_relevance" for issue in result.issues)


def test_tag_issue_uses_the_problem_tag_as_evidence():
    result = QCAgent(MockLLMClient()).check(
        "빈티지 체크셔츠 출근룩",
        make_idea(),
        make_post("빈티지 체크셔츠로 출근 코디를 소개합니다.", tags=["스타일링팁"]),
    )

    issue = next(issue for issue in result.issues if issue.category == "tag_relevance")
    assert issue.evidence == "스타일링팁"


def test_generic_image_only_plan_is_reported_for_specific_source():
    post = make_post("빈티지 체크셔츠 출근룩을 소개합니다.", image_text="기부할 의류를 상자에 담는 모습")
    result = QCAgent(MockLLMClient()).check("빈티지 체크셔츠 출근룩", make_idea(), post)

    assert result.status == "NEEDS_REVISION"
    assert any(issue.category == "image_relevance" for issue in result.issues)


def test_image_issue_uses_the_image_plan_evidence():
    post = make_post(
        "빈티지 체크셔츠 출근룩을 소개합니다.",
        image_text="기부할 의류를 상자에 담는 모습",
    )
    result = QCAgent(MockLLMClient()).check("빈티지 체크셔츠 출근룩", make_idea(), post)

    issue = next(issue for issue in result.issues if issue.category == "image_relevance")
    assert issue.evidence == "기부할 의류를 상자에 담는 모습"


def test_organization_article_accepts_classification_and_reuse_images():
    source = "안 입는 옷을 정리해서 기부할 옷과 재활용할 옷을 분류하고 싶다."
    post = make_post(
        "남길 옷과 기부할 옷을 분류합니다.",
        tags=["안입는옷정리", "옷재활용"],
        image_text="남길 옷과 기부할 옷을 분류하는 장면",
    )
    post.title = "안 입는 옷 정리와 기부"
    post.keyword = "안 입는 옷 정리"
    post.image_plan.images.append(ImagePrompt(
        placement="section-2",
        purpose="기부할 옷을 상자에 담는 모습",
        prompt="기부할 옷을 상자에 담는 모습",
        image_type="reuse",
        alt_text="기부할 옷을 상자에 담는 모습",
    ))
    result = QCAgent(MockLLMClient()).check(source, make_idea(), post)

    assert not any(issue.category == "image_relevance" for issue in result.issues)

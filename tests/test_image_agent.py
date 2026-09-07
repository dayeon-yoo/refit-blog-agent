from agents.image_agent import ImageAgent
from models.schemas import BlogPost


def make_post(content: str) -> BlogPost:
    return BlogPost(
        title="가을 옷장 정리와 재사용",
        keyword="가을 옷장 정리",
        summary="옷을 분류하고 상태를 확인해 다시 활용하는 방법",
        content=content,
        cta="작은 부분부터 시작해 보세요.",
        tags=["가을옷장정리", "옷재사용"],
    )


def test_image_sections_keep_article_order_and_have_meaningful_placements():
    post = make_post(
        """가을 옷장을 열면 작년에 입던 옷이 한가득 나옵니다.

## 옷을 어떻게 나눌까요?
자주 입는 옷과 잘 입지 않는 옷을 두 그룹으로 나눠보세요.

## 정리 전후를 비교해보세요
정리 전 옷장과 정리 후 옷장의 차이를 비교하면 변화를 쉽게 확인할 수 있습니다.

## 안 입는 셔츠를 다시 활용하는 방법
안 입는 셔츠를 새로운 소품으로 리폼해 재사용할 수 있습니다.

## 같은 옷도 다르게 입을 수 있을까요?
셔츠와 니트를 조합해 일상 코디를 만들어봅니다.

## 마무리
오늘부터 작은 부분부터 시작해 보세요."""
    )

    planned = ImageAgent().plan(post)
    images = planned.image_plan.images

    assert 1 <= len(images) <= 5
    placements = [int(image.placement.split("-")[-1]) for image in images if image.placement.startswith("section-")]
    assert placements == sorted(placements)
    assert any(image.image_type == "classification" for image in images)
    assert any(image.image_type == "comparison" for image in images)
    assert any(image.image_type == "reuse" for image in images)
    assert any(image.image_type == "styling" for image in images)
    assert all(image.placement != "body" for image in images)


def test_visual_prompts_describe_scenes_instead_of_copying_body_sentences():
    sentence = "자주 입는 옷과 잘 입지 않는 옷을 나눠보세요."
    post = make_post(f"## 분류 기준\n{sentence}")

    image = ImageAgent().plan(post).image_plan.images[0]

    assert image.image_type == "classification"
    assert sentence not in image.prompt
    assert "두 그룹" in image.prompt
    assert "옷장" in image.prompt or "가정" in image.prompt
    assert "자주 입는 옷" in image.alt_text
    assert "검색" not in image.alt_text


def test_short_generic_article_does_not_get_filled_with_generic_images():
    post = make_post("오늘은 옷에 대한 짧은 생각을 적어봅니다.")

    planned = ImageAgent().plan(post)

    assert planned.image_plan is not None
    assert planned.image_plan.images == []


def test_headingless_content_is_supported():
    post = make_post("옷장 앞에서 자주 입는 옷과 잘 입지 않는 옷을 분류하고 정리하는 방법을 소개합니다.")

    planned = ImageAgent().plan(post)

    assert planned.image_plan is not None
    assert planned.image_plan.images
    assert planned.image_plan.images[0].placement == "hero-intro"


def test_repeated_image_types_are_limited():
    post = make_post(
        """## 정리 방법 1
옷장 안의 옷을 정리하는 방법입니다.
## 정리 방법 2
계절 옷을 정리하고 수납하는 방법입니다.
## 정리 방법 3
서랍 속 옷을 정리하는 방법입니다.
## 정리 방법 4
행거의 옷을 정리하는 방법입니다.
## 정리 방법 5
보관할 옷을 정리하는 방법입니다."""
    )

    images = ImageAgent().plan(post).image_plan.images

    assert len(images) <= 5
    assert max((sum(image.image_type == image_type for image in images) for image_type in {
        image.image_type for image in images
    }), default=0) <= 2


def test_repeated_detail_images_have_distinct_visual_purposes_and_prompts():
    post = make_post(
        """## 옷의 상태 확인
얼룩과 늘어남이 있는 옷은 가까이 살펴보고 보관 여부를 판단합니다.

## 소재와 봉제 확인
니트의 원단 결, 봉제선, 라벨과 단추 상태를 확인합니다."""
    )

    images = ImageAgent().plan(post).image_plan.images
    detail_images = [image for image in images if image.image_type == "detail"]

    assert len(detail_images) == 2
    assert len({image.purpose for image in detail_images}) == 2
    assert len({image.prompt for image in detail_images}) == 2
    assert len({image.alt_text for image in detail_images}) == 2
    assert "얼룩" in detail_images[0].prompt
    assert "봉제" in detail_images[1].prompt

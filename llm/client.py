from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        raise NotImplementedError

    @abstractmethod
    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def __init__(self, response_factory: Callable[[str, Type[BaseModel], Dict[str, Any]], Any] | None = None):
        self.response_factory = response_factory
        self.calls: List[Dict[str, Any]] = []

    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        if not payload:
            raise ValueError(f"Empty payload for schema {schema.__name__}")
        self.calls.append({"prompt": prompt, "schema": schema, "payload": payload})
        if self.response_factory is not None:
            response = self.response_factory(prompt, schema, payload)
        elif schema.__name__ == "BlogPost":
            response = self._default_blog_post(payload)
        elif schema.__name__ == "IdeaExpansionResult":
            response = self._default_idea_expansion(payload)
        elif schema.__name__ == "BlogIdea" and "selected_candidate" in payload:
            response = self._default_refined_idea(payload)
        else:
            response = payload
        return response if isinstance(response, schema) else schema.model_validate(response)

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        if not payloads:
            return []
        return [schema.model_validate(item) for item in payloads]

    @staticmethod
    def _default_blog_post(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a topic-neutral response for local tests without an API call."""
        title = str(payload.get("title", "")).strip()
        keyword = str(payload.get("keyword", "")).strip()
        angle = str(payload.get("angle", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        count = payload.get("required_item_count")
        rifit_connection = str(payload.get("rifit_connection", "")).strip()

        intro = angle or summary or keyword or title
        connection_text = rifit_connection.lower()
        circulation_connection = any(
            marker in connection_text
            for marker in ("순환", "재사용", "재활용", "수거", "기부", "처분", "처리", "지속 가능한")
        )
        sections = [
            f"## {title}",
            f"{title}을 고민하다 보면, 막상 어디서부터 시작해야 할지 망설여질 때가 있는데요."
            f" {intro}를 기준으로 지금 할 수 있는 작은 방법부터 살펴보겠습니다.",
        ]
        if isinstance(count, int) and count > 0:
            for index in range(1, count + 1):
                lead = [
                    "먼저 현재 상태를 가볍게 확인해 보세요.",
                    "이 단계에서는 준비물을 많이 늘리기보다 집에 있는 것을 먼저 살펴보면 좋습니다.",
                    "실제로 해보면 생각보다 간단한 부분부터 손이 가는데요.",
                ][(index - 1) % 3]
                sections.extend([
                    f"## {index}. {title} 핵심 항목 {index}",
                    f"{lead} {intro}와 연결된 {index}번째 방법은 현재 상황에 맞게 순서를 조절해 적용해 보세요.",
                ])
        else:
            for index in range(1, 4):
                sections.extend([
                    f"## {title}을 시작하기 전에 확인할 점 {index}",
                    f"{intro}를 살펴볼 때는 한 번에 모두 바꾸려 하지 말고, {index}번째로 눈에 들어오는 부분부터 확인해 보세요."
                    " 준비물과 현재 상태를 먼저 점검하면 시행착오를 줄일 수 있습니다.",
                ])
        if circulation_connection:
            sections.extend([
                "## 더 이상 입지 않는 옷은 어떻게 순환시킬까요?",
                "더 이상 손이 가지 않는 옷은 바로 버리기보다 상태와 사용 가능성을 먼저 살펴보세요. "
                "다시 입거나 수선할 수 있는 옷은 활용하고, 계속 보관할 이유가 없다면 기부·수거·재활용처럼 "
                "상태에 맞는 순환 방법을 정해 옷의 다음 쓰임으로 연결해보는 것이 좋습니다.",
            ])
        cta = "오늘은 가장 부담 없는 항목 하나만 골라 실제로 해보세요."
        if circulation_connection:
            cta = "입지 않는 옷이 생겼다면 상태를 확인한 뒤 재사용이나 순환 방법을 한 가지 정해보세요."
        sections.append(cta)
        return {
            "title": title,
            "keyword": keyword,
            "content": "\n\n".join(sections),
            "summary": summary or f"{title}에 대한 실천 방법을 정리했습니다.",
            "cta": cta,
        }

    @staticmethod
    def _default_idea_expansion(payload: Dict[str, Any]) -> Dict[str, Any]:
        source = str(payload.get("user_input", "")).strip()
        context = MockLLMClient._extract_idea_context(source)
        count = min(5, max(3, int(payload.get("candidate_count", 3))))
        topic = context["subject"]
        if context["experience"]:
            if context["clothing"]:
                directions = [
                    ("구매 경험", "경험 후기", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 직접 사보니 알게 된 점"),
                    ("선택 기준", "구매 가이드", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 고를 때 확인한 기준"),
                    ("실제 착용", "스타일링 실험", f"{topic} 하나로 출근 코디를 바꿔본 기록"),
                    ("매력 탐구", "배경 탐구", f"{topic}{MockLLMClient._particle(topic, '이', '가')} 오래 입고 싶은 옷이 되는 이유"),
                    ("새 제품과 비교", "비교 후기", f"새 체크 셔츠와 {topic}, 직접 입어보니 달랐던 점"),
                ]
            else:
                directions = [
                    ("직접 경험", "경험 후기", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 직접 해보니 알게 된 점"),
                    ("선택 기준", "실용 가이드", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 선택할 때 확인한 기준"),
                    ("변화 과정", "사례 기록", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 시작하고 달라진 점"),
                    ("비교 관찰", "비교", f"{topic}{MockLLMClient._particle(topic, '을', '를')} 하기 전과 후에 달랐던 점"),
                    ("이유 탐구", "원인 분석", f"왜 {topic}에 관심을 갖게 되었을까"),
                ]
        elif context["trend"]:
            colors = context["materials"] or [topic]
            first = colors[0]
            second = colors[1] if len(colors) > 1 else topic
            joined = "·".join(item.replace("코어", "") for item in colors[:3])
            directions = [
                ("트렌드 흐름", "트렌드 분석", f"{first} 다음은 {second}? 컬러 트렌드가 바뀌는 방식"),
                ("유행 소비 속도", "원인 분석", f"{first}에서 {second}로, 컬러 유행은 왜 빨리 바뀔까"),
                ("실제 선택", "스타일링 가이드", f"{joined}, 지금 옷장에 들이기 좋은 컬러는"),
                ("코어 트렌드 문화", "소비문화 분석", f"{joined}, 새로운 코어 컬러는 어떻게 만들어질까"),
                ("기존 아이템 활용", "스타일링 실전 가이드", f"유행 지난 {first} 아이템을 다시 입는 방법"),
            ]
        elif context["question"]:
            directions = [
                ("사실 확인", "팩트체크", f"{topic}, 실제로 어떤 과정을 거칠까"),
                ("과정 추적", "과정 해설", f"{topic}이 처리되는 과정을 따라가봤다"),
                ("문제점 점검", "문제 해결", f"{topic}에서 소비자가 놓치기 쉬운 점"),
                ("소비자 관점", "소비문화 분석", f"{topic}을 둘러싼 선택과 책임"),
                ("실천 가이드", "체크리스트", f"{topic}을 확인하기 전에 알아둘 것"),
            ]
        elif context["collection"]:
            directions = [
                ("분류 기준", "정보 가이드", "의류수거함에 넣어도 될까? 헷갈리는 옷 분류 기준"),
                ("보내는 방법 비교", "비교 가이드", "의류수거함과 기부, 안 입는 옷은 어디로 보내야 할까?"),
                ("기부 전 확인", "체크리스트", "기부하기 좋은 옷은 따로 있을까? 보내기 전 확인할 것"),
                ("수거 전후 과정", "과정 설명", "헌옷은 모두 의류수거함에 넣으면 될까?"),
                ("이동 과정", "과정 해설", "의류수거함에 넣은 옷은 그다음 어디로 갈까?"),
            ]
        elif context["shopping"]:
            directions = [
                ("찾아보는 기준", "쇼핑 가이드", f"{topic}에서 원하는 아이템을 찾을 때 볼 포인트"),
                ("아이템 관찰", "스타일 탐구", f"{topic}에서 발견하는 옷의 매력"),
                ("선택 기준", "구매 가이드", f"{topic} 아이템을 고를 때 확인할 기준"),
                ("코디 활용", "스타일링 가이드", f"빈티지 옷으로 {topic} 무드 연출하기"),
                ("실패 줄이기", "체크리스트", f"{topic} 쇼핑 전에 확인하면 좋은 항목"),
            ]
        else:
            directions = [
                ("관찰과 정보", "에세이", f"{topic}을 둘러싼 관찰과 새롭게 알게 된 점"),
                ("선택 기준", "실용 가이드", f"{topic}을 시작하기 전에 확인할 기준"),
                ("비교 관찰", "비교", f"{topic}과 다른 선택지를 비교해보니"),
                ("문제 해결", "문제 해결", f"{topic}에서 자주 생기는 문제와 해결 방법"),
                ("배경 탐구", "배경 분석", f"{topic}이 사람들의 관심을 끄는 이유"),
            ]
        candidates = []
        for index in range(count):
            perspective, content_format, title = directions[index]
            question = context["questions"][index % len(context["questions"])]
            brief = MockLLMClient._expansion_brief(topic, perspective, content_format, context)
            candidates.append({
                "candidate_id": f"candidate-{index + 1}",
                "title": title,
                "perspective": perspective,
                "content_format": content_format,
                "key_question": question,
                "brief_description": brief,
            })
        return {"source_input": source, "candidates": candidates}

    @staticmethod
    def _particle(value: str, with_final: str, without_final: str) -> str:
        for char in reversed(value.strip()):
            if "가" <= char <= "힣":
                has_final = (ord(char) - ord("가")) % 28 != 0
                return with_final if has_final else without_final
        return without_final

    @staticmethod
    def _expansion_brief(topic: str, perspective: str, content_format: str, context: Dict[str, Any]) -> str:
        topic_object = context.get("object_subject", f"{topic}{MockLLMClient._particle(topic, '을', '를')}")
        topic_subject = f"{topic}{MockLLMClient._particle(topic, '이', '가')}"
        experience_detail = context.get("experience_detail", "")
        observation = context.get("observation", "")
        if content_format == "경험 후기":
            suffix = f" {experience_detail}" if experience_detail else ""
            return f"{topic_object} 직접 선택하고 사용해본 과정에서 느낀 점과 예상 밖의 경험을 기록합니다.{suffix}"
        if content_format == "구매 가이드":
            return f"{topic_object} 고를 때 살펴본 상태, 핏, 가격 같은 실제 선택 기준을 정리합니다."
        if content_format == "스타일링 실험":
            return f"{topic_object} 일상적인 옷과 조합해본 코디와 착용 후 변화를 보여줍니다."
        if content_format == "배경 탐구":
            return f"{topic_subject} 오래 입고 싶은 대상으로 느껴지는 소재와 디자인의 이유를 살펴봅니다."
        if content_format == "비교 후기":
            return f"새 제품과 {topic_object} 직접 비교해 착용감, 상태, 분위기의 차이를 정리합니다."
        if content_format == "트렌드 분석":
            return f"{observation or topic}에서 보이는 변화가 어떤 순서로 주목받는지 실제 사례를 따라가 봅니다."
        if content_format == "원인 분석":
            return f"{observation or topic}의 관심이 이동하는 배경을 소비 방식과 미디어 흐름에서 살펴봅니다."
        if content_format == "스타일링 가이드":
            return f"{topic} 중 지금 활용하기 좋은 색을 옷장 속 기본 아이템과 조합하는 방법을 제안합니다."
        if content_format == "소비문화 분석":
            return f"{topic} 같은 이름의 유행이 만들어지고 교체되는 과정을 소비문화 관점에서 설명합니다."
        if content_format == "스타일링 How-to":
            return f"유행이 지난 {topic_object} 다른 옷과 조합해 지금의 스타일로 다시 활용하는 방법을 보여줍니다."
        if content_format == "팩트체크":
            return f"{topic}에 대해 알려진 정보와 실제 처리 과정을 나누어 확인합니다."
        if content_format == "과정 해설":
            return f"{topic}이 수거된 뒤 어떤 단계를 거치는지 처음부터 끝까지 따라갑니다."
        if content_format == "문제 해결":
            return f"{topic_object} 둘러싼 소비자의 오해와 실제로 확인할 수 있는 해결 방법을 정리합니다."
        if content_format == "체크리스트":
            return f"{topic_object} 확인하거나 선택하기 전에 소비자가 점검할 항목을 구체적으로 제시합니다."
        return f"{topic_object} {perspective} 관점에서 살펴보고 독자가 참고할 구체적인 장면과 기준을 제시합니다."

    @staticmethod
    def _extract_idea_context(source: str) -> Dict[str, Any]:
        """Extract lightweight editorial signals without a topic-specific fixture."""
        cleaned = " ".join(source.split())
        notes = re.findall(r"[\[(]([^\])]+)[\])]", cleaned)
        base = re.sub(r"[\[(][^\])]+[\])]", "", cleaned).strip()
        if ":" in base:
            prefix, remainder = base.split(":", 1)
            if any(marker in prefix.lower() for marker in ("일기", "기록", "메모", "관찰")):
                base = remainder.strip()
        base = re.sub(r"\s+(?:내가|직접).*$", "", base).strip(" -")
        base = re.sub(r"\s+(?:에서\s+)?찾기$", "", base).strip()
        base = re.sub(r"의\s*(?:세계|이야기|기록)$", "", base).strip()

        observation = ""
        claim = ""
        if any(marker in cleaned for marker in ("같아", "것 같", "느껴", "보여", "손이 덜", "고민")):
            claim = cleaned.rstrip(".!?。？！")
        if "에서" in cleaned and any(marker in cleaned for marker in ("로", "으로", "대신", "보다", "나", "vs")):
            observation = cleaned.rstrip(".!?。？！")
        elif claim:
            observation = claim
        question_text = cleaned.rstrip(".!?。？！") if ("?" in cleaned or "정말" in cleaned or "일까" in cleaned or "될까" in cleaned) else ""
        contrast = any(marker in cleaned.lower() for marker in ("대신", "보다", "반면", " vs ", "에서", "나"))
        proposed_direction = any(marker in cleaned for marker in ("토대로", "중심으로", "연결", "해보고", "소개"))

        tokens = re.findall(r"[가-힣A-Za-z0-9]+", base.lower())
        particles = ("으로", "에서", "에게", "까지", "부터", "처럼", "보다", "과", "와", "나", "고", "은", "는", "이", "가", "을", "를", "의", "에", "로")
        meaningful = []
        for token in tokens:
            for particle in particles:
                if token.endswith(particle) and len(token) - len(particle) >= 2:
                    token = token[: -len(particle)]
                    break
            if len(token) > 1 and token not in {"요즘", "정말", "같아", "것", "새롭게"} and token not in meaningful:
                meaningful.append(token)

        materials = [token for token in meaningful if token.endswith(("코어", "색"))]
        clothing = any(term in base for term in ("옷", "셔츠", "바지", "니트", "의류", "코디", "패션", "스타일"))
        experience = bool(notes) or any(marker in cleaned for marker in ("직접", "경험", "산 경험", "사서", "입어", "해봤"))
        trend = bool(materials) or any(marker in cleaned for marker in ("유행", "뜨는", "끝나", "트렌드", "코어"))
        question = "?" in cleaned or any(marker in cleaned for marker in ("정말", "왜", "어떻게", "될까", "일까"))
        collection = any(marker in cleaned for marker in ("의류수거함", "수거함", "기부", "수거", "헌옷"))
        shopping = any(marker in cleaned for marker in ("쇼핑", "찾기", "구매", "고르"))
        if materials:
            subject = "·".join(materials[:3])
        else:
            subject = re.split(r"\s+(?:정말|과연|왜|어떻게)", base, maxsplit=1)[0].strip()
            subject = re.sub(r"(은|는|이|가)\s*$", "", subject).strip()
            subject = subject or base or "이 주제"
        if notes and clothing and not materials:
            subject = re.sub(r"\s+", " ", base).strip()
        object_subject = f"{subject}{MockLLMClient._particle(subject, '을', '를')}"
        questions = [
            f"{object_subject} 실제로 어떻게 살펴볼 수 있을까?",
            f"{subject}에서 독자가 확인할 점은 무엇일까?",
            f"{object_subject} 다른 선택지와 비교하면 무엇이 다를까?",
        ]
        descriptions = [
            "{topic}에 관한 {perspective} 글로, 사용자의 관찰과 구체적인 장면을 함께 정리합니다.",
            "{topic_object} 중심으로 실제 사례와 선택 기준을 살펴보고 독자가 적용할 단서를 제시합니다.",
            "{topic}에 얽힌 경험과 질문을 바탕으로 한 편의 독립적인 콘텐츠 흐름을 구성합니다.",
        ]
        return {
            "subject": subject,
            "object_subject": object_subject,
            "materials": materials,
            "clothing": clothing,
            "experience": experience,
            "trend": trend,
            "question": question,
            "questions": questions,
            "descriptions": descriptions,
            "observation": observation,
            "claim": claim,
            "question_text": question_text,
            "contrast": contrast,
            "proposed_direction": proposed_direction,
            "experience_detail": " ".join(notes).strip(),
            "collection": collection,
            "shopping": shopping,
        }

    @staticmethod
    def _default_refined_idea(payload: Dict[str, Any]) -> Dict[str, Any]:
        source = str(payload.get("source_input", "")).strip()
        candidate = payload.get("selected_candidate", {})
        title = str(candidate.get("title", source)).strip()
        perspective = str(candidate.get("perspective", "문제 해결")).strip()
        content_format = str(candidate.get("content_format", "실용 가이드")).strip()
        question = str(candidate.get("key_question", "독자는 무엇을 확인해야 할까요?")).strip()
        description = str(candidate.get("brief_description", source)).strip()
        revision = str(payload.get("revision_request", "")).strip()
        revision_note = f" 추가 요청은 다음과 같이 반영합니다: {revision}" if revision else ""
        return {
            "title": title,
            "keyword": " ".join(title.split()[:4]),
            "search_intent": "정보 탐색",
            "angle": f"{perspective} 관점의 {content_format}으로 {question}",
            "rifit_connection": "사용자가 가진 옷과 소비 습관을 지속 가능한 순환으로 연결하는 실천을 안내합니다.",
            "seasonality": 0.5,
            "summary": f"{description}{revision_note}",
            "content_format": content_format,
            "content_perspective": perspective,
            "key_question": question,
            "target_reader": "이 변화의 의미와 다음 행동을 알고 싶은 독자",
            "outline": [
                f"{topic}의 현재 상황과 독자가 주목할 변화" if (topic := source.rstrip(".!?。？！")) else "원본 아이디어의 현재 상황",
                f"{perspective} 관점에서 살펴볼 핵심 장면",
                f"{content_format} 형식으로 비교하거나 확인할 기준",
                "독자가 자신의 상황에 적용할 수 있는 구체적인 예시",
                "다음 행동으로 이어지는 정리와 실천 팁",
            ],
        }


class LocalFallbackLLM(MockLLMClient):
    pass


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when MOCK_MODE=false")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        if not payload:
            raise ValueError(f"Empty payload for schema {schema.__name__}")
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=schema,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("OpenAI returned no parsed result for the schema")
            return parsed
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed for schema {schema.__name__}: {exc}") from exc

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        if not payloads:
            return []
        return [self.generate_structured(prompt, schema, payload) for payload in payloads]


def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.mock_mode:
        return MockLLMClient()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing while MOCK_MODE=false")
    return OpenAILLMClient(api_key=settings.openai_api_key, model=settings.model_name)

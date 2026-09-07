from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from llm.client import LLMClient, get_llm_client
from config.settings import get_settings
from models.schemas import BlogIdea
import math


def _tokens(text: str) -> Set[str]:
    # simple whitespace-based tokens, remove short tokens
    return {t for t in text.lower().split() if len(t) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    uni = a | b
    return len(inter) / len(uni)


class IdeaGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "idea.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def _fallback_candidates(self) -> List[dict]:
        return [
            {
                "title": "의류 정리와 순환, 현실적인 1가지 기준",
                "keyword": "의류 순환 방법",
                "search_intent": "정보 탐색",
                "angle": "정리 기준과 순환 경로를 함께 고려하는 실용적인 의류 관리 지침",
                "rifit_connection": "의류 정리와 수거의 자연스러운 연결",
                "seasonality": 0.85,
            },
            {
                "title": "안 입는 반팔 처리, 이렇게 정리하면 실속 있다",
                "keyword": "안 입는 반팔 처리",
                "search_intent": "문제 해결",
                "angle": "상태별 분류와 재사용 가능성을 함께 보는 실전 처리 팁",
                "rifit_connection": "헌옷 수거와 의류 재사용 관점으로 연결",
                "seasonality": 0.9,
            },
            {
                "title": "옷장 정리 시즌별 팁, 의류 순환까지 이어가기",
                "keyword": "옷장 정리 시즌별 팁",
                "search_intent": "정보 탐색",
                "angle": "계절 전환 시점에 필요한 의류 분류와 순환 전략",
                "rifit_connection": "계절별 의류 정리와 헌옷 수거를 함께 설계",
                "seasonality": 0.92,
            },
        ]

    def generate(self, trend_results, seo_results, brand_context=None) -> List[BlogIdea]:
        prompt = self.prompt
        settings = get_settings()
        idea_count = max(1, settings.idea_count)

        if not trend_results and not seo_results:
            # return fallback candidates sized to idea_count
            base = self._fallback_candidates()
            candidates = (base * ((idea_count // len(base)) + 1))[:idea_count]
            return self.client.generate_many(prompt, BlogIdea, candidates)

        # build a seed pool from trends and seo keywords (preserve order, unique)
        seeds = []
        if trend_results:
            for t in trend_results:
                if t.topic not in seeds:
                    seeds.append(t.topic)
        if seo_results:
            for s in seo_results:
                if s.keyword not in seeds:
                    seeds.append(s.keyword)

        # editorial perspectives to diversify idea topics (no seed injection into title)
        perspectives = [
            ("옷장 정리", "옷장 정리 - 공간 구성과 분류 기준, 정리 루틴 중심의 실용적 접근"),
            ("의류 관리", "의류 관리 - 소재별 세탁·보관·수선 관점의 실무 조언"),
            ("헌옷 처리", "헌옷 처리 - 기부·수거·정리 단계에서의 실용 가이드"),
            ("의류 재활용", "재활용/업사이클링 - 재료 분리와 재활용 아이디어 소개"),
            ("재사용", "재사용 - 집에서 다시 쓰기 가능한 아이템 선정과 준비 방법"),
            ("중고 거래", "중고 판매/리셀 - 사진·설명·가격 책정 팁과 플랫폼별 전략"),
            ("빈티지", "빈티지 스타일 - 리폼으로 가치 올리기와 스타일 연출"),
            ("수선/리폼", "수선·리폼 - 간단 수선으로 수명 늘리는 방법과 비용 가이드"),
            ("소비 습관", "소비 습관 - 장기적 의류 소비 절약과 계획 방법"),
            ("환경/순환", "환경과 의류 순환 - 지역 자원과 기부 연결 방법 및 영향"),
        ]

        # natural title templates per perspective — avoid English tokens and machiney patterns
        templates = {
            "옷장 정리": [
                "지금 당장 옷장 한 칸 정리로 생활을 가볍게 만드는 방법",
                "옷장 정리를 빠르게 끝내는 실전 루틴"
            ],
            "의류 관리": [
                "소재별로 알아보는 세탁·보관의 기본",
                "자주 입는 옷 오래 입히는 작은 습관"
            ],
            "헌옷 처리": [
                "헌옷 보낼 때 이것만은 꼭 확인하세요",
                "헌옷을 기부하거나 재사용할 때 실수하지 않는 법"
            ],
            "의류 재활용": [
                "버려진 옷을 새롭게 바꾸는 재활용 아이디어",
                "작은 수선으로 만드는 업사이클링 사례"
            ],
            "재사용": [
                "집에서 바로 재사용할 수 있는 의류 정리법",
                "다음 시즌까지 입을 옷을 고르는 기준"
            ],
            "중고 거래": [
                "중고로 잘 팔리는 사진과 설명의 비결",
                "리셀 전 꼭 확인할 체크포인트"
            ],
            "빈티지": [
                "빈티지 무드 내는 간단한 리폼 아이디어",
                "오래된 옷을 스타일로 살리는 방법"
            ],
            "수선/리폼": [
                "집에서 시도해볼 수 있는 쉬운 옷 수선 가이드",
                "수선 비용과 효과를 비교해보는 방법"
            ],
            "소비 습관": [
                "옷 구매 빈도를 줄이는 실천법",
                "필요한 옷만 남기는 소비 계획 세우기"
            ],
            "환경/순환": [
                "동네에서 할 수 있는 의류 순환 참여 방법",
                "의류 순환이 환경에 주는 영향과 시작 방법"
            ],
        }

        candidates = []
        seen_titles = set()
        seen_token_sets: List[Set[str]] = []

        # iterate seeds and perspectives to generate candidates; ensure intra-run diversity
        s_index = 0
        p_index = 0
        while len(candidates) < idea_count and (s_index < len(seeds)):
            seed = seeds[s_index]
            perspective = perspectives[p_index % len(perspectives)]
            key = perspective[0]
            angle_desc = perspective[1]
            tmpl_list = templates.get(key, [templates[list(templates.keys())[0]][0]])
            tmpl = tmpl_list[(s_index + p_index) % len(tmpl_list)]

            # produce title without forcing seed inclusion
            title = tmpl

            # avoid duplicate or overly similar titles within this run
            tok = _tokens(title)
            too_similar = False
            for prev in seen_token_sets:
                if _jaccard(tok, prev) > 0.45:
                    too_similar = True
                    break
            if too_similar:
                # advance perspective to try a different angle
                p_index += 1
                # if we've cycled through many perspectives, fallback to add variant with seed as suffix phrase
                if p_index - s_index > len(perspectives) * 2:
                    title = f"{tmpl} — {seed}과 연결된 실전 방법"
                    tok = _tokens(title)
                    too_similar = False

            if title in seen_titles:
                # avoid identical title
                p_index += 1
                s_index += 1
                continue

            # pick keyword and seasonality from matching seo or trend
            keyword = seed
            search_intent = "정보 탐색"
            seasonality = 0.5
            if seo_results:
                for s in seo_results:
                    if s.keyword == seed:
                        keyword = s.keyword
                        search_intent = s.search_intent
                        seasonality = round(min(max(s.seasonality, 0.0), 1.0), 2)
                        break
            if trend_results and seasonality == 0.5:
                for t in trend_results:
                    if t.topic == seed:
                        seasonality = round(min(max(t.relevance_score, 0.0), 1.0), 2)
                        break

            candidates.append(
                {
                    "title": title,
                    "keyword": keyword,
                    "search_intent": search_intent,
                    "angle": angle_desc,
                    "rifit_connection": "의류 순환/수거/재사용과 연결되는 실용적 콘텐츠",
                    "seasonality": seasonality,
                }
            )

            seen_titles.add(title)
            seen_token_sets.append(tok)

            # advance indices
            p_index += 1
            if p_index % len(perspectives) == 0:
                s_index += 1

        # if still short, fill with non-seed generic variants
        i = 1
        while len(candidates) < idea_count:
            title = f"의류 순환을 일상에 적용하는 실용적 방법 {i}"
            if title not in seen_titles:
                candidates.append({
                    "title": title,
                    "keyword": "의류 순환 방법",
                    "search_intent": "정보 탐색",
                    "angle": "생활 속에서 바로 적용 가능한 의류 재사용 실전 팁",
                    "rifit_connection": "의류 정리와 수거의 자연스러운 연결",
                    "seasonality": 0.5,
                })
                seen_titles.add(title)
            i += 1

        return self.client.generate_many(prompt, BlogIdea, candidates[:idea_count])

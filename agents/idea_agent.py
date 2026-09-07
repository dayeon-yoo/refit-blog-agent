from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from config.settings import get_settings
from models.schemas import BlogIdea


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

        # editorial perspectives to diversify idea types
        perspectives = [
            ("wardrobe", "실용적인 옷장 정리/관리와 적용 방법"),
            ("reuse", "헌옷 처리와 재사용 경로 안내"),
            ("recycle", "의류 수거와 재활용/업사이클링 소개"),
            ("resale", "중고 거래/세컨핸드 판매 전략"),
            ("vintage", "빈티지 스타일과 리폼 아이디어"),
            ("repair", "수선/리폼/업사이클링 실전 가이드"),
            ("habit", "지속 가능한 소비 습관과 의류 소비의 대안"),
            ("care", "실용적인 의류 관리와 세탁/보관 팁"),
        ]

        # title templates per perspective
        templates = {
            "wardrobe": ["{seed}로 옷장 정리할 때 꼭 확인할 5가지", "{seed}을(를) 바로 정리하는 현실적 가이드"],
            "reuse": ["{seed}을(를) 기부하거나 재사용할 때 체크리스트", "{seed} 재사용 전 꼭 확인할 점"],
            "recycle": ["{seed}을(를) 재활용/업사이클링으로 연결하는 방법", "{seed} 재활용 아이디어: 이렇게 활용하세요"],
            "resale": ["{seed} 판매 전 알아둘 리셀 팁", "{seed}을(를) 중고로 높은 가격에 파는 법"],
            "vintage": ["{seed}에서 찾은 빈티지 스타일 적용법", "{seed} 리폼으로 빈티지 무드 만들기"],
            "repair": ["{seed} 수선으로 오래 입는 법", "{seed}을(를) 집에서 손쉽게 수선하는 방법"],
            "habit": ["{seed}으로 시작하는 지속 가능한 옷 소비 습관", "{seed}을 통해 소비를 줄이는 실천법"],
            "care": ["{seed} 소재별 세탁·건조 체크리스트", "{seed} 보관 전 꼭 확인할 사항"],
        }

        candidates = []
        seen_titles = set()

        # interleave seeds with perspectives to produce diverse candidates
        p_count = len(perspectives)
        p_index = 0
        s_index = 0
        while len(candidates) < idea_count and (s_index < len(seeds)):
            seed = seeds[s_index]
            perspective_key, perspective_desc = perspectives[p_index % p_count]
            tmpl = templates[perspective_key][(s_index + p_index) % len(templates[perspective_key])]
            title = tmpl.format(seed=seed)
            if title in seen_titles:
                # try alternate template or skip
                alt = templates[perspective_key][0].format(seed=seed)
                title = alt
            seen_titles.add(title)

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
                    "angle": perspective_desc,
                    "rifit_connection": "의류 순환/수거/재사용과 연결되는 실용적 콘텐츠",
                    "seasonality": seasonality,
                }
            )

            # advance perspective and seed indexes
            p_index += 1
            if p_index % p_count == 0:
                s_index += 1

        # if still short, fill with sensible variants
        i = 1
        while len(candidates) < idea_count:
            title = f"의류 재사용과 순환: 실용 팁 {i}"
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

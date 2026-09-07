from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.schemas import BlogPost, TagRecommendation, TagRecommendationResult


class KeywordDataProvider(ABC):
    """Provider abstraction for keyword metrics."""

    @abstractmethod
    def fetch_metrics(self, keywords: List[str]) -> Dict[str, Dict]:
        raise NotImplementedError()


@dataclass(frozen=True)
class KeywordMetrics:
    """Provider-neutral keyword metrics used by the ranking layer."""

    keyword: str
    search_volume: Optional[int] = None
    competition: Optional[float] = None
    publishing_volume: Optional[int] = None
    saturation: Optional[float] = None
    metrics_available: bool = False

    @classmethod
    def from_mapping(cls, keyword: str, values: Optional[Dict]) -> "KeywordMetrics":
        values = values or {}
        return cls(
            keyword=keyword,
            search_volume=values.get("search_volume"),
            competition=values.get("competition"),
            publishing_volume=values.get("publishing_volume"),
            saturation=values.get("saturation"),
            metrics_available=values.get(
                "metrics_available",
                any(values.get(key) is not None for key in (
                    "search_volume", "competition", "publishing_volume", "saturation"
                )),
            ),
        )


class MockKeywordDataProvider(KeywordDataProvider):
    """Deterministic local provider used without external keyword APIs."""

    def __init__(self):
        self.was_called = False
        self.requested_keywords: List[str] = []

    def _hash_to_range(self, s: str, a: int, b: int) -> int:
        h = 0
        for ch in s:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return a + (h % (b - a + 1))

    def _hash_to_float(self, s: str) -> float:
        h = 0
        for ch in s:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return (h % 1000) / 1000.0

    def fetch_metrics(self, keywords: List[str]) -> Dict[str, Dict]:
        self.was_called = True
        self.requested_keywords = list(keywords)
        out: Dict[str, Dict] = {}
        for kw in keywords:
            out[kw] = {
                "search_volume": self._hash_to_range(kw, 50, 20000),
                "competition": round(self._hash_to_float(kw), 3),
                "publishing_volume": self._hash_to_range(kw + "pub", 0, 500),
                "saturation": round(self._hash_to_float(kw + "sat"), 3),
            }
        return out


class TagAgent:
    """Generate and rank natural, space-free blog tag candidates."""

    RELEVANCE_WEIGHT = 0.42
    SEARCH_VOLUME_WEIGHT = 0.10
    COMPETITION_WEIGHT = 0.12
    SATURATION_WEIGHT = 0.12
    PUBLISHING_VOLUME_WEIGHT = 0.04
    CONTENT_COVERAGE_WEIGHT = 0.10
    LONGTAIL_WEIGHT = 0.04
    DIVERSITY_WEIGHT = 0.06
    MIN_RELEVANCE = 0.22

    _TOKEN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9_-]*")
    _HEADING = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.MULTILINE)
    _PUNCTUATION = re.compile(r"[,.!?+/:;@#$%^&*\\()\[\]<>|\"']")

    _BLOCKED_WORDS = {
        "대한", "새로운", "다양한", "중요한", "즐거운", "실용적인", "방법을",
        "정리하고", "찾아가는", "아이템을", "시작과", "스타일을", "옷장을",
        "하는", "할", "해야", "있습니다", "있어요", "그리고", "하지만", "찾기",
        "발견", "즐거움", "기존", "조합하는", "재조합할",
    }
    _STOPWORDS = {
        "이", "가", "은", "는", "을", "를", "의", "에", "로", "와", "과", "도",
        "만", "에서", "에게", "까지", "부터", "처럼", "보다", "위해", "통해",
        "것", "수", "때", "점", "경우", "정도", "위한", "대한",
    }
    _BAD_ENDINGS = (
        "으로써", "으로서", "에게서", "에게로", "하고", "하며", "면서", "지만",
        "하는", "찾아가는", "정리하고", "시작과", "아이템을", "방법을",
        "스타일을", "옷장을",
    )
    _GENERIC_SINGLE_WORDS = {
        "가을", "겨울", "봄", "여름", "계절", "시즌", "정리", "방법", "스타일",
        "옷", "의류", "옷장", "패션", "코디", "스타일링", "아이템", "정보", "팁", "노하우", "활용", "새활용",
    }
    _COMMON_SUFFIXES = ("방법", "팁", "노하우")
    _TOPIC_SUFFIXES = {
        "정리": ("방법", "팁", "노하우"),
        "보관": ("방법", "팁", "노하우"),
        "코디": ("방법", "팁"),
        "스타일링": ("방법", "팁"),
        "활용": ("방법", "팁"),
        "업사이클링": ("방법", "팁"),
        "재활용": ("방법", "팁"),
        "기부": ("방법", "팁"),
        "수거": ("방법", "팁"),
        "판매": ("방법", "팁"),
        "처리": ("방법", "팁"),
    }
    _ACTION_TERMS = {"기부", "수거", "판매", "폐기", "처리"}
    _SYNONYMS = {"의류": "옷", "의복": "옷", "의상": "옷"}
    _CONCEPT_TERMS = {
        "가을", "겨울", "봄", "여름", "계절", "시즌", "옷", "의류", "헌옷", "옷장",
        "셔츠", "니트", "청바지", "티셔츠", "스웨터", "가방", "패션", "정리", "보관",
        "코디", "스타일", "스타일링", "활용", "업사이클링", "리폼", "재활용", "재사용",
        "재조합", "조합", "기부", "수거", "판매", "처리", "폐기", "청소", "비우기",
        "안입는옷", "아이템", "방법", "팁", "노하우", "수납", "공간", "분류", "기준",
        "루틴", "주기", "지속", "가능한", "지속가능", "소비", "구매", "충동구매",
        "소재", "브랜드", "윤리", "용도", "관리", "착용", "오래", "처분", "순환",
    }
    _METADATA = {
        "seo", "검색의도", "검색 의도", "ai", "agent", "prompt", "workflow",
        "metadata", "브랜드평가", "브랜드 평가", "fit_score", "critic", "critic score",
        "콘텐츠생성과정",
    }

    def __init__(self, provider: KeywordDataProvider, max_candidates: int = 100):
        self.provider = provider
        self.max_candidates = max_candidates

    @staticmethod
    def _normalize_space(value: str) -> str:
        return " ".join((value or "").lower().strip().split())

    def _clean_phrase(self, value: str) -> str:
        value = self._normalize_space(value)
        value = self._PUNCTUATION.sub(" ", value)
        return " ".join(value.split()).strip(" -:;.")

    def _is_bad_term(self, term: str) -> bool:
        term = self._clean_phrase(term)
        if not term or term in self._BLOCKED_WORDS or term in self._STOPWORDS:
            return True
        if len(term) < 2:
            return True
        if all(ord(ch) < 128 for ch in term) and len(term) < 3:
            return True
        if any(term.endswith(ending) for ending in self._BAD_ENDINGS):
            return True
        # Short Korean words such as "가을" can end with a particle-like
        # syllable, so apply this heuristic only to longer tokens.
        return len(term) >= 3 and term.endswith(("을", "를", "은", "는", "의", "에", "로", "와", "과"))

    def _is_likely_noun(self, token: str) -> bool:
        """Conservative heuristic for phrase construction without an NLP dependency."""
        token = self._clean_phrase(token)
        if self._is_bad_term(token):
            return False
        if self._PUNCTUATION.search(token):
            return False
        if any(word in token for word in ("어떻게", "할까", "해야할지", "어디서")):
            return False
        return True

    def _phrase_terms(self, phrase: str) -> List[str]:
        terms: List[str] = []
        for raw in self._clean_phrase(phrase).split():
            normalized = self._normalize_known_form(raw)
            if normalized and self._is_likely_noun(normalized) and normalized not in terms:
                terms.append(self._SYNONYMS.get(normalized, normalized))
        return terms

    def _normalize_known_form(self, raw: str) -> str:
        raw = self._clean_phrase(raw)
        if not raw:
            return ""
        if self._is_likely_noun(raw):
            return raw
        particles = ("으로써", "으로서", "에게서", "에게로", "하고", "하며", "면서", "으로", "로", "을", "를", "은", "는", "의", "에", "와", "과")
        for concept in sorted(self._CONCEPT_TERMS, key=len, reverse=True):
            for particle in particles:
                if raw == concept + particle:
                    return concept
        return ""

    def _term_sequences(self, phrase: str) -> List[List[str]]:
        sequences: List[List[str]] = []
        current: List[str] = []
        for raw in self._clean_phrase(phrase).split():
            normalized = self._normalize_known_form(raw)
            if normalized and (normalized in self._CONCEPT_TERMS or all(ord(ch) < 128 for ch in normalized)):
                current.append(self._SYNONYMS.get(normalized, normalized))
            elif current:
                sequences.append(current)
                current = []
        if current:
            sequences.append(current)
        return sequences

    def _concept_terms(self, phrase: str) -> List[str]:
        terms = self._phrase_terms(phrase)
        return [
            term for term in terms
            if term in self._CONCEPT_TERMS or all(ord(ch) < 128 for ch in term)
        ]

    def _to_tag_form(self, raw: str) -> str:
        phrase = self._clean_phrase(raw)
        if not phrase:
            return ""
        terms = phrase.split()
        if not terms or any(not self._is_likely_noun(term) for term in terms):
            return ""
        return "".join(terms)

    def _contains_english_templates(self, value: str) -> bool:
        low = value.lower()
        return any(token in low for token in ("how to", "guide", "tips", "problem", "alternatives", "howto"))

    def _topic_text(self, post: BlogPost) -> str:
        headings = " ".join(self._HEADING.findall(post.content or ""))
        return " ".join(
            self._normalize_space(value)
            for value in (post.keyword, post.title, post.summary, headings)
            if value
        )

    def _primary_season(self, post: BlogPost) -> Optional[str]:
        """Select the season with the strongest signal in high-priority fields."""
        weights = (
            (post.keyword or "", 4),
            (post.title or "", 3),
            (post.summary or "", 2),
            (" ".join(self._HEADING.findall(post.content or "")), 1),
        )
        scores: Dict[str, int] = {}
        for text, weight in weights:
            for season in ("봄", "여름", "가을", "겨울"):
                if season in text:
                    scores[season] = scores.get(season, 0) + weight
        return max(scores, key=scores.get) if scores else None

    def _has_korean_topic(self, post: BlogPost) -> bool:
        priority_text = " ".join((post.keyword or "", post.title or "", post.summary or ""))
        return bool(re.search(r"[가-힣]", priority_text))

    def _is_english_only(self, tag: str) -> bool:
        return bool(re.fullmatch(r"[a-z0-9]+", tag.lower()))

    def _body_concepts(self, post: BlogPost) -> List[str]:
        """Extract known concepts and explicit action phrases from body text."""
        text = (post.content or "").lower()
        concepts = self._content_terms(post)
        phrase_rules = (
            (("안 입", "안입", "입지 않", "한 번도 입"), "안입는옷"),
            (("분류", "나눠", "나누"), "분류"),
            (("수납", "수납공간"), "수납"),
            (("정기적으로", "주기적으로", "루틴", "주기"), "루틴"),
            (("공간을 활용", "공간 활용"), "공간활용"),
        )
        for markers, concept in phrase_rules:
            if not any(marker in text for marker in markers) or concept in concepts:
                continue
            # A single incidental mention should not create a large family of
            # 안입는옷 tags when the article is actually about another topic.
            if concept == "안입는옷":
                marker_count = sum(text.count(marker) for marker in markers)
                priority_text = self._topic_text(post)
                if marker_count < 2 and not any(marker in priority_text for marker in markers):
                    continue
            concepts.append(concept)
        return concepts

    def _content_terms(self, post: BlogPost) -> List[str]:
        """Use body text only for terms that are already topic-shaped or known concepts."""
        trusted = set(self._concept_terms(self._topic_text(post)))
        known = {
            "옷", "옷장", "의류", "헌옷", "정리", "보관", "코디", "스타일", "스타일링",
            "활용", "업사이클링", "재활용", "재사용", "기부", "수거", "판매", "처리",
            "계절", "가을", "겨울", "봄", "여름", "시즌", "가방", "셔츠", "니트",
            "지속", "가능한", "소비", "구매", "소재", "브랜드", "용도", "관리", "착용",
            "오래", "처분", "순환", "충동구매",
        }
        body_terms: List[str] = []
        for raw in self._TOKEN.findall((post.content or "").lower()):
            term = self._SYNONYMS.get(raw, raw)
            if term in trusted or term in known:
                if self._is_likely_noun(term) and term not in body_terms:
                    body_terms.append(term)
        return body_terms

    def _candidate_is_topic_relevant(self, tag: str, post: BlogPost) -> bool:
        return self._relevance(tag, post) > 0.0

    def is_valid_tag(self, tag: str, post: BlogPost) -> bool:
        """Validate the final representation, which must be one space-free search term."""
        if not tag or tag != tag.strip() or any(ch.isspace() for ch in tag):
            return False
        if len(tag) < 2 or len(tag) > 60:
            return False
        if self._PUNCTUATION.search(tag) or self._contains_english_templates(tag):
            return False

        low = tag.lower()
        if any(meta in low for meta in self._METADATA):
            return False
        if low in self._BLOCKED_WORDS or self._is_bad_term(low):
            return False
        if any(question in low for question in ("어떻게", "할까", "해야", "어디", "어떤")):
            return False
        if self._is_sentence_like(tag):
            return False
        if len(tag) <= 3 and low in self._GENERIC_SINGLE_WORDS:
            return False
        if self._has_korean_topic(post) and self._is_english_only(tag):
            # Keep an exact English keyword phrase only when it is the source
            # keyword; do not let English single words fill Korean results.
            if tag != self._to_tag_form(post.keyword or ""):
                return False
        return self._candidate_is_topic_relevant(tag, post)

    def _is_sentence_like(self, tag: str) -> bool:
        if any(tag.endswith(ending) for ending in self._BAD_ENDINGS):
            return True
        return tag in self._BLOCKED_WORDS

    def _relevance(self, tag: str, post: BlogPost) -> float:
        """Weight keyword/title more heavily than accidental body occurrences."""
        compact = tag.lower()

        def field_score(value: str) -> float:
            terms = self._concept_terms(value)
            if not terms:
                return 0.0
            hits = sum(1 for term in terms if term in compact)
            exact = self._to_tag_form(value)
            score = min(1.0, hits / max(1, min(3, len(terms))))
            if exact and exact == compact:
                score = 1.0
            return score

        headings = " ".join(self._HEADING.findall(post.content or ""))
        score = (
            0.60 * field_score(post.keyword or "")
            + 0.25 * field_score(post.title or "")
            + 0.10 * field_score(post.summary or "")
            + 0.05 * field_score(headings)
        )
        if score == 0.0:
            body_terms = self._content_terms(post)
            if any(term in compact for term in body_terms):
                return 0.05
        return min(1.0, score)

    def _add_candidate(self, raw: str, post: BlogPost, candidates: List[str], seen: set[str]) -> None:
        tag = self._to_tag_form(raw)
        if not tag or tag in seen:
            return
        if not self.is_valid_tag(tag, post):
            return
        seen.add(tag)
        candidates.append(tag)

    def _add_ngrams(self, terms: List[str], post: BlogPost, candidates: List[str], seen: set[str]) -> None:
        clean = [term for term in terms if self._is_likely_noun(term)]
        for size in (3, 2):
            for index in range(len(clean) - size + 1):
                self._add_candidate(" ".join(clean[index : index + size]), post, candidates, seen)

    def _add_sequence_ngrams(self, phrase: str, post: BlogPost, candidates: List[str], seen: set[str]) -> None:
        for sequence in self._term_sequences(phrase):
            self._add_ngrams(sequence, post, candidates, seen)

    def _add_suffixes(self, bases: List[str], topic_text: str, post: BlogPost, candidates: List[str], seen: set[str]) -> None:
        for base in bases:
            if any(base.endswith(suffix) for suffix in self._COMMON_SUFFIXES):
                continue
            suffixes: Tuple[str, ...] = ()
            for stem, allowed in self._TOPIC_SUFFIXES.items():
                if stem in base and stem in topic_text:
                    suffixes = allowed
                    break
            if not suffixes:
                continue
            for suffix in suffixes:
                self._add_candidate(base + suffix, post, candidates, seen)

    def _semantic_group(self, tag: str) -> str:
        for intent in ("수납", "분류", "기준", "루틴", "주기", "스타일링", "스타일", "코디", "활용", "정리", "보관", "기부", "수거", "판매", "처리", "재활용", "재사용", "업사이클링"):
            if intent in tag:
                return intent
        return "핵심"

    def _context_adjustment(self, tag: str, post: BlogPost) -> float:
        adjustment = 0.0
        primary = self._primary_season(post)
        seasons = ("봄", "여름", "가을", "겨울")
        if primary and primary in tag:
            adjustment += 0.10
        elif primary and any(season in tag for season in seasons if season != primary):
            adjustment -= 0.14
        if self._has_korean_topic(post) and self._is_english_only(tag):
            adjustment -= 0.15
        return adjustment

    def _deduplicate_similar(self, candidates: List[str], limit: int) -> List[str]:
        """Limit variants from one concept group while preserving useful diversity."""
        suffixes = ("노하우", "방법", "팁", "추천", "정보")

        def root(tag: str) -> str:
            changed = True
            while changed:
                changed = False
                for suffix in suffixes:
                    if tag.endswith(suffix) and len(tag) > len(suffix):
                        tag = tag[: -len(suffix)]
                        changed = True
                        break
            return tag

        grouped: Dict[str, List[str]] = {}
        for candidate in candidates:
            grouped.setdefault(root(candidate), []).append(candidate)

        selected: List[str] = []
        # Keep at most three variants per root in the final candidate pool.
        for group in grouped.values():
            for candidate in group[:3]:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= limit:
                    return selected
        return selected

    def generate_candidates(self, post: BlogPost, target: int = 80) -> List[str]:
        target = min(max(target, 0), self.max_candidates)
        if target == 0:
            return []

        title_terms = self._concept_terms(post.title or "")
        keyword_terms = self._concept_terms(post.keyword or "")
        summary_terms = self._concept_terms(post.summary or "")
        heading_text = " ".join(self._HEADING.findall(post.content or ""))
        heading_terms = self._concept_terms(heading_text)
        topic_text = self._topic_text(post)

        candidates: List[str] = []
        seen: set[str] = set()

        # Preserve high-signal search phrases first.
        for phrase in (post.keyword or "", post.title or "", post.summary or ""):
            self._add_candidate(phrase, post, candidates, seen)

        # Then create only short, topic-shaped combinations.
        self._add_sequence_ngrams(post.keyword or "", post, candidates, seen)
        self._add_sequence_ngrams(post.title or "", post, candidates, seen)
        self._add_sequence_ngrams(post.summary or "", post, candidates, seen)
        self._add_sequence_ngrams(heading_text, post, candidates, seen)

        priority_terms = keyword_terms + title_terms + summary_terms + heading_terms
        body_concepts = self._body_concepts(post)
        concept_terms: List[str] = []
        for term in priority_terms + body_concepts:
            if term not in concept_terms:
                concept_terms.append(term)

        # Single terms are a fallback, and generic Korean words are excluded.
        for term in keyword_terms + title_terms:
            if term not in self._GENERIC_SINGLE_WORDS:
                self._add_candidate(term, post, candidates, seen)

        # Add directional combinations such as 가을+옷장+정리, not every permutation.
        seasons = [self._primary_season(post)] if self._primary_season(post) else []
        items = [term for term in concept_terms if term in {"옷", "의류", "헌옷", "안입는옷", "옷장", "셔츠", "니트", "청바지", "티셔츠", "스웨터"}]
        intent_vocabulary = {"정리", "보관", "코디", "스타일", "스타일링", "활용", "업사이클링", "재활용", "재사용", "기부", "수거", "판매", "처리", "폐기", "청소", "비우기", "재조합", "조합", "수납", "공간활용", "분류", "기준", "루틴", "주기"}
        priority_actions = set(priority_terms).intersection(self._ACTION_TERMS)
        intents = [
            term
            for term in priority_terms + body_concepts
            if term in intent_vocabulary and (term not in self._ACTION_TERMS or term in priority_actions)
        ]
        for season in seasons:
            for item in items:
                self._add_candidate(season + item, post, candidates, seen)
                for intent in intents:
                    self._add_candidate(season + item + intent, post, candidates, seen)
                    if intent in {"스타일", "스타일링", "코디"}:
                        self._add_candidate(season + intent, post, candidates, seen)
        for item in items:
            for intent in intents:
                self._add_candidate(item + intent, post, candidates, seen)

        # Turn explicit body concepts into search-shaped phrases instead of
        # treating the body words as arbitrary single-token candidates.
        body_templates = {
            "안입는옷": ("안입는옷정리", "안입는옷활용"),
            "분류": ("계절옷분류", "옷정리기준"),
            "기준": ("옷정리기준", "옷장정리기준"),
            "수납": ("옷장수납", "계절옷수납", "옷수납방법"),
            "공간활용": ("옷장공간활용",),
            "루틴": ("옷장정리루틴", "옷장정리주기"),
        }
        for concept in body_concepts:
            for template in body_templates.get(concept, ()):
                self._add_candidate(template, post, candidates, seen)

        # Suffixes are allowed only for an already relevant base concept.
        self._add_suffixes(list(candidates), topic_text, post, candidates, seen)

        return self._deduplicate_similar(candidates, target)

    def _normalize_metric(self, value, values):
        if not values or value is None:
            return None
        low, high = min(values), max(values)
        return 0.5 if low == high else (value - low) / (high - low)

    def _content_coverage(self, tag: str, post: BlogPost) -> float:
        """Measure how many high-signal content fields support a candidate."""
        headings = " ".join(self._HEADING.findall(post.content or ""))
        fields = (
            (post.keyword or "", 0.45),
            (post.title or "", 0.25),
            (post.summary or "", 0.15),
            (headings, 0.10),
            (" ".join(self._body_concepts(post)), 0.05),
        )
        coverage = 0.0
        for value, weight in fields:
            terms = self._concept_terms(value)
            if terms and any(term in tag.lower() for term in terms):
                coverage += weight
        return min(1.0, coverage)

    def _longtail_score(self, tag: str, post: BlogPost) -> float:
        """Prefer specific multi-concept phrases without rewarding sentence fragments."""
        terms = [term for term in self._concept_terms(self._topic_text(post)) if term in tag]
        if self._is_english_only(tag):
            return 0.1 if len(tag) >= 10 else 0.0
        if len(terms) >= 3:
            return 1.0
        if len(terms) == 2 or len(tag) >= 6:
            return 0.75
        return 0.25

    def _effective_relevance(self, tag: str, post: BlogPost) -> float:
        relevance = self._relevance(tag, post)
        primary = self._primary_season(post)
        if primary and primary in tag:
            relevance += 0.12
        elif primary and any(season in tag for season in ("봄", "여름", "가을", "겨울") if season != primary):
            relevance -= 0.12
        return max(0.0, min(1.0, relevance))

    def recommend(self, post: BlogPost, candidate_target: int = 80, final_k: int = 30) -> TagRecommendationResult:
        final_k = min(max(final_k, 0), 30)
        candidates = self.generate_candidates(post, target=candidate_target)
        if not candidates or final_k == 0:
            post.tags = []
            return TagRecommendationResult(tags=[])

        # Only validated, deduplicated candidates reach the provider.
        candidates = [candidate for candidate in candidates if self.is_valid_tag(candidate, post)]
        if not candidates:
            post.tags = []
            return TagRecommendationResult(tags=[])

        try:
            raw_metrics = self.provider.fetch_metrics(candidates)
        except Exception as exc:
            partial_metrics = getattr(exc, "partial_metrics", None)
            if partial_metrics is None:
                raise
            raw_metrics = partial_metrics
        metric_models = {
            candidate: KeywordMetrics.from_mapping(candidate, raw_metrics.get(candidate))
            for candidate in candidates
        }
        entries: List[Tuple[str, float, KeywordMetrics]] = [
            (candidate, self._effective_relevance(candidate, post), metric_models[candidate])
            for candidate in candidates
        ]
        # A high-volume generic word cannot compensate for weak topical fit.
        entries = [entry for entry in entries if entry[1] >= self.MIN_RELEVANCE]
        if not entries:
            post.tags = []
            return TagRecommendationResult(tags=[])

        volumes = [entry[2].search_volume for entry in entries if entry[2].search_volume is not None]
        pubs = [entry[2].publishing_volume for entry in entries if entry[2].publishing_volume is not None]
        comps = [entry[2].competition for entry in entries if entry[2].competition is not None]
        sats = [entry[2].saturation for entry in entries if entry[2].saturation is not None]

        scored: List[TagRecommendation] = []
        for tag, relevance, metric in entries:
            n_volume = self._normalize_metric(metric.search_volume, volumes)
            n_pub = self._normalize_metric(metric.publishing_volume, pubs)
            n_comp = self._normalize_metric(metric.competition, comps)
            n_sat = self._normalize_metric(metric.saturation, sats)

            # Missing provider values are neutral rather than rewarding or
            # penalizing a candidate. Relevance remains the largest component.
            demand = 0.5 if n_volume is None else n_volume
            publishing = 0.5 if n_pub is None else n_pub
            competition = 0.5 if n_comp is None else 1.0 - n_comp
            saturation = 0.5 if n_sat is None else 1.0 - n_sat
            coverage = self._content_coverage(tag, post)
            longtail = self._longtail_score(tag, post)
            ranking = (
                self.RELEVANCE_WEIGHT * relevance
                + self.SEARCH_VOLUME_WEIGHT * demand
                + self.COMPETITION_WEIGHT * competition
                + self.SATURATION_WEIGHT * saturation
                + self.PUBLISHING_VOLUME_WEIGHT * (1.0 - publishing)
                + self.CONTENT_COVERAGE_WEIGHT * coverage
                + self.LONGTAIL_WEIGHT * longtail
            )
            ranking = max(0.0, min(1.0, ranking))
            scored.append(TagRecommendation(
                tag=tag,
                relevance_score=round(relevance, 3),
                search_volume=metric.search_volume,
                competition=metric.competition,
                publishing_volume=metric.publishing_volume,
                saturation=metric.saturation,
                metrics_available=metric.metrics_available,
                ranking_score=round(ranking, 3),
            ))

        scored.sort(key=lambda item: item.ranking_score, reverse=True)
        out_tags: List[TagRecommendation] = []
        seen: set[str] = set()
        root_counts: Dict[str, int] = {}
        group_counts: Dict[str, int] = {}
        remaining = list(scored)
        while remaining:
            recommendation = max(
                remaining,
                key=lambda item: (
                    item.ranking_score
                    + self.DIVERSITY_WEIGHT / (1 + group_counts.get(self._semantic_group(item.tag), 0))
                    - 0.04 * group_counts.get(self._semantic_group(item.tag), 0)
                ),
            )
            remaining.remove(recommendation)
            if recommendation.tag in seen:
                continue
            root = recommendation.tag
            changed = True
            while changed:
                changed = False
                for suffix in ("노하우", "방법", "팁", "추천", "정보"):
                    if root.endswith(suffix) and len(root) > len(suffix):
                        root = root[: -len(suffix)]
                        changed = True
                        break
            if root_counts.get(root, 0) >= 3:
                continue
            group = self._semantic_group(recommendation.tag)
            if group_counts.get(group, 0) >= 8:
                continue
            seen.add(recommendation.tag)
            root_counts[root] = root_counts.get(root, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            recommendation.ranking_score = round(
                max(
                    0.0,
                    min(
                        1.0,
                        recommendation.ranking_score
                        + self.DIVERSITY_WEIGHT / group_counts[group]
                        - 0.04 * (group_counts[group] - 1),
                    ),
                ),
                3,
            )
            out_tags.append(recommendation)
            if len(out_tags) >= final_k:
                break

        out_tags.sort(key=lambda item: item.ranking_score, reverse=True)

        post.tags = [recommendation.tag for recommendation in out_tags]
        return TagRecommendationResult(tags=out_tags)

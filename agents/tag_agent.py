from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import math

from models.schemas import (
    BlogPost,
    TagRecommendation,
    TagRecommendationResult,
)


class KeywordDataProvider(ABC):
    """Abstract provider interface for keyword metrics.

    Implementations should return a mapping from keyword -> metrics dict.
    Metrics keys (optional): `search_volume` (int), `competition` (0-1 float),
    `publishing_volume` (int), `saturation` (0-1 float).
    """

    @abstractmethod
    def fetch_metrics(self, keywords: List[str]) -> Dict[str, Dict]:
        raise NotImplementedError()


class MockKeywordDataProvider(KeywordDataProvider):
    """Deterministic mock provider that returns repeatable pseudo-metrics.

    This mock does not call any external service. Values are derived from the
    keyword text so tests are deterministic.
    """

    def __init__(self):
        self.was_called = False

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
        out: Dict[str, Dict] = {}
        for kw in keywords:
            sv = self._hash_to_range(kw, 50, 20000)
            comp = round(self._hash_to_float(kw), 3)
            pub = self._hash_to_range(kw + "pub", 0, 500)
            sat = round(self._hash_to_float(kw + "sat"), 3)
            out[kw] = {
                "search_volume": sv,
                "competition": comp,
                "publishing_volume": pub,
                "saturation": sat,
            }
        return out


class TagAgent:
    def __init__(self, provider: KeywordDataProvider, max_candidates: int = 100):
        self.provider = provider
        self.max_candidates = max_candidates

    def _normalize(self, s: str) -> str:
        return " ".join(s.lower().strip().split())

    def _ngrams(self, words: List[str], n: int) -> List[str]:
        return [" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]

    def generate_candidates(self, post: BlogPost, target: int = 80) -> List[str]:
        parts: List[str] = []
        title = self._normalize(post.title)
        keyword = self._normalize(post.keyword)
        summary = self._normalize(post.summary or "")
        content = self._normalize(post.content or "")

        # Base: explicit keyword and title
        if keyword:
            parts.append(keyword)
        if title and title != keyword:
            parts.append(title)

        # n-grams from title and summary (1-3 grams)
        for text in (title, summary):
            words = [w for w in text.split() if len(w) > 1]
            for n in (1, 2, 3):
                parts.extend(self._ngrams(words, n))

        # Method/problem phrasing
        if keyword:
            parts.extend([
                f"how to {keyword}",
                f"{keyword} tips",
                f"{keyword} guide",
                f"{keyword} problems",
                f"{keyword} vs alternatives",
            ])

        # Derive related broader terms from keyword tokens
        kw_tokens = keyword.split()
        if len(kw_tokens) > 1:
            parts.append(kw_tokens[-1])
            parts.append(" ".join(kw_tokens[:-1]))

        # Short phrases from content: first 40 words window
        content_words = [w for w in content.split() if len(w) > 1]
        for i in range(min(10, len(content_words))):
            n = 2
            if i + n <= len(content_words):
                parts.append(" ".join(content_words[i : i + n]))

        # Clean, unique and limit
        seen = set()
        cleaned: List[str] = []
        for p in parts:
            tag = self._normalize(p)
            if not tag:
                continue
            if tag in seen:
                continue
            # simple length/filter heuristics
            if len(tag) < 2 or len(tag) > 80:
                continue
            seen.add(tag)
            cleaned.append(tag)
            if len(cleaned) >= self.max_candidates:
                break

        # If too few, include simple tokens from title/keyword
        if len(cleaned) < target:
            extras = list({t for t in (title + " " + keyword).split() if len(t) > 2})
            for e in extras:
                if e not in seen:
                    cleaned.append(e)
                    seen.add(e)
                if len(cleaned) >= target:
                    break

        return cleaned[: max(target, 0)]

    def _relevance(self, tag: str, post: BlogPost) -> float:
        # Simple overlap heuristic between tag and title/summary/keyword
        t = set(tag.split())
        title = set(self._normalize(post.title).split())
        summary = set(self._normalize(post.summary or "").split())
        keyword = set(self._normalize(post.keyword).split())
        score = 0.0
        if t & keyword:
            score += 0.6
        if t & title:
            score += 0.25
        if t & summary:
            score += 0.15
        # normalize to 0..1
        return min(1.0, score)

    def recommend(self, post: BlogPost, candidate_target: int = 80, final_k: int = 30) -> TagRecommendationResult:
        candidates = self.generate_candidates(post, target=candidate_target)
        if not candidates:
            return TagRecommendationResult(tags=[])

        metrics = self.provider.fetch_metrics(candidates)

        # build initial entries
        entries: List[Tuple[str, float, Dict]] = []
        for c in candidates:
            rel = self._relevance(c, post)
            m = metrics.get(c, {})
            entries.append((c, rel, m))

        # filter low relevance
        entries = [e for e in entries if e[1] > 0.05]

        if not entries:
            return TagRecommendationResult(tags=[])

        # normalize numeric metrics across entries
        volumes = [e[2].get("search_volume") for e in entries if e[2].get("search_volume") is not None]
        pubs = [e[2].get("publishing_volume") for e in entries if e[2].get("publishing_volume") is not None]
        comps = [e[2].get("competition") for e in entries if e[2].get("competition") is not None]
        sats = [e[2].get("saturation") for e in entries if e[2].get("saturation") is not None]

        def norm(v, arr):
            if not arr or v is None:
                return None
            mn = min(arr)
            mx = max(arr)
            if mx == mn:
                return 0.5
            return (v - mn) / (mx - mn)

        scored: List[TagRecommendation] = []
        for tag, rel, m in entries:
            sv = m.get("search_volume")
            comp = m.get("competition")
            pub = m.get("publishing_volume")
            sat = m.get("saturation")

            n_sv = norm(sv, volumes)
            n_comp = norm(comp, comps)
            n_pub = norm(pub, pubs)
            n_sat = norm(sat, sats)

            # base ranking: favor relevance most
            # Only include metric components if present
            score = 0.0
            weight_total = 0.0
            # relevance weight
            score += 0.6 * rel
            weight_total += 0.6
            if n_sv is not None:
                score += 0.2 * (n_sv)
                weight_total += 0.2
            if n_comp is not None:
                score += 0.1 * (1.0 - n_comp)
                weight_total += 0.1
            if n_pub is not None:
                score += 0.05 * (1.0 - n_pub)
                weight_total += 0.05
            if n_sat is not None:
                score += 0.05 * (1.0 - n_sat)
                weight_total += 0.05

            # normalize to 0..1 by weight_total
            ranking = score / weight_total if weight_total > 0 else rel
            ranking = max(0.0, min(1.0, ranking))

            scored.append(
                TagRecommendation(
                    tag=tag,
                    relevance_score=round(rel, 3),
                    search_volume=sv,
                    competition=comp,
                    publishing_volume=pub,
                    saturation=sat,
                    ranking_score=round(ranking, 3),
                )
            )

        # sort by ranking_score desc, remove duplicates, limit to final_k
        scored.sort(key=lambda x: x.ranking_score, reverse=True)
        out_tags: List[TagRecommendation] = []
        seen = set()
        for s in scored:
            if s.tag in seen:
                continue
            # filter out obvious mismatch: relevance nearly zero
            if s.relevance_score < 0.05:
                continue
            seen.add(s.tag)
            out_tags.append(s)
            if len(out_tags) >= final_k:
                break

        # attach simple tag strings to post.tags (do not expose internal metrics)
        post.tags = [t.tag for t in out_tags]

        return TagRecommendationResult(tags=out_tags)

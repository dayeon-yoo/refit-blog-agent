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

    def _clean_tag_raw(self, s: str) -> str:
        """Clean punctuation and unwanted characters from a raw candidate."""
        # keep Hangul, ascii letters/numbers and spaces
        import re

        if not s:
            return ""
        # normalize whitespace
        s = " ".join(s.split())
        # remove common english template tokens punctuation
        s = re.sub(r"[,!/\\+\(\)\[\]<>\|\?\"]+", " ", s)
        # strip stray punctuation
        s = s.strip(" -:;,." )
        # remove common trailing Korean particle suffixes when attached
        s = re.sub(r"(.*?)(으로써|으로서|으로|로|에서|에게|에게서|에게로|에|으로서)$", r"\1", s)
        return s

    def _to_tag_form(self, s: str) -> str:
        """Convert a cleaned phrase into a tag-friendly normalized form.

        Strategy: remove short particle tokens and join remaining tokens without spaces
        to produce compact noun-phrase-like tags (e.g., '옷장 정리' -> '옷장정리').
        Avoid producing empty results.
        """
        if not s:
            return ""
        tokens = [t for t in s.split() if t]
        # remove obvious single-character Korean particles
        particles = {"이", "가", "은", "는", "을", "를", "에", "의", "로", "와", "과", "도", "만", "께", "에서"}
        filtered = [t for t in tokens if not (len(t) == 1 and t in particles)]
        if not filtered:
            filtered = tokens
        # join tokens to form compact tag
        tag = "".join(filtered)
        return tag

    def _is_malformed_korean(self, s: str) -> bool:
        bad_subs = ["은(는)", "을(를)", "이(가)", "하는법", "방법은", "정리을", "재사용을"]
        low = s.lower()
        for b in bad_subs:
            if b in low:
                return True
        return False

    def _is_likely_noun(self, token: str) -> bool:
        """Relaxed heuristic: accept multi-character tokens that are not obviously
        sentence fragments or english function words. We allow noun-phrases and
        compound words commonly used as tags (e.g., '옷장정리', '가을옷정리',
        '옷정리방법'). Reject very short tokens or strings with sentence markers.
        """
        if not token:
            return False
        low = token.lower()
        # too short
        if len(low) < 2:
            return False
        # obvious question/verb fragments that look like sentences
        bad_endings = ("어떻게", "해야", "해야할까", "할까", "해야할지", "할지")
        for be in bad_endings:
            if low.endswith(be) or be in low:
                return False
        # disallow pure punctuation or tokens containing punctuation
        import re

        if re.search(r"[\,\.!\?\+/:;@#\$%\^&\*\\()\[\]<>\|]", low):
            return False
        # allow longer ascii words (>=4) as potential english keywords
        if all(ord(c) < 128 for c in token):
            return len(token) >= 4
        return True

    def _contains_english_templates(self, s: str) -> bool:
        low = s.lower()
        bad = ["how to", "guide", "tips", "problem", "problems", "vs", "alternatives", "howto"]
        for b in bad:
            if b in low:
                return True
        return False

    def is_valid_tag(self, tag: str, post: BlogPost) -> bool:
        """Validate a normalized tag string for suitability.

        - Not empty, reasonable length
        - No english template remnants
        - No malformed Korean markers
        - Not sentence-like or containing internal metadata
        - Related to the post at a minimal level (token overlap)
        """
        if not tag:
            return False
        # basic length checks
        if len(tag) < 2 or len(tag) > 60:
            return False
        # disallow punctuation
        import re

        if re.search(r"[\,\.!\?\+/:;@#\$%\^&\*\\\(\)\[\]<>\|]", tag):
            return False

        if self._contains_english_templates(tag):
            return False
        if self._is_malformed_korean(tag):
            return False

        low = tag.lower()
        # internal metadata blocklist
        metas = ["seo", "검색의도", "검색 의도", "ai", "agent", "prompt", "브랜드평가", "content 생성", "콘텐츠생성"]
        for m in metas:
            if m in low:
                return False

        # avoid question-like tags
        questions = ["어떻게", "할까", "해야", "어디", "어떤"]
        for q in questions:
            if q in low:
                return False

        # minimal relatedness: require tag to contain at least one meaningful token
        raw_key = self._normalize((post.title or "") + " " + (post.keyword or "") + " " + (post.summary or "") + " " + (post.content or ""))
        key_tokens = [k for k in raw_key.split() if len(k) > 1]
        stopwords = {"지금", "당장", "방법론", "생활", "것", "이", "가", "을", "를", "에", "의", "로", "한", "수", "중"}
        key_tokens = [k for k in key_tokens if k not in stopwords]
        if not key_tokens:
            return False

        # if tag contains any key token, accept (covers noun-phrases and compound forms)
        for kt in key_tokens:
            if kt and kt in low:
                return True

        # allow if shares a meaningful 2-char substring with title/keyword (fallback)
        t = self._normalize((post.title or "") + " " + (post.keyword or ""))
        for i in range(max(0, len(t) - 1)):
            sub = t[i : i + 2]
            if sub and sub in low and sub not in stopwords:
                return True

        return False

    def _ngrams(self, words: List[str], n: int) -> List[str]:
        return [" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]

    def generate_candidates(self, post: BlogPost, target: int = 80) -> List[str]:
        # Hierarchical, combination-based candidate generation
        title = self._normalize(post.title or "")
        keyword = self._normalize(post.keyword or "")
        summary = self._normalize(post.summary or "")
        content = self._normalize(post.content or "")

        text_blob = " ".join([title, keyword, summary, content])

        # Helper: extract tokens (words of length>1)
        def tokens_from(text: str):
            return [w for w in text.split() if len(w) > 1]

        title_tokens = tokens_from(title)
        keyword_tokens = tokens_from(keyword)
        summary_tokens = tokens_from(summary)
        content_tokens = tokens_from(content)

        all_tokens = list(dict.fromkeys(title_tokens + keyword_tokens + summary_tokens + content_tokens))

        # synonym map to prefer natural Korean words
        syn = {"의류": "옷", "의복": "옷", "의상": "옷", "의류관리": "옷관리"}

        # core nouns (bases) that are likely to form tags
        # preserve token order (title/keyword/summary/content) and dedupe
        core_bases = []
        def push_core(x: str):
            if not x:
                return
            if x not in core_bases:
                core_bases.append(x)

        for t in all_tokens:
            t_clean = self._clean_tag_raw(t)
            if not t_clean:
                continue
            mapped = syn.get(t_clean, t_clean)
            push_core(mapped)

        # ensure explicit keyword tokens appear first
        for t in keyword_tokens:
            if t:
                push_core(syn.get(t, t))

        # predefined concept stems to consider when present in text
        concept_stems = ["정리", "기부", "재활용", "중고", "수거", "판매", "처리", "보관", "재사용", "순환"]

        present_stems = [s for s in concept_stems if s in text_blob]

        candidates: List[str] = []
        seen = set()

        def add_candidate(raw: str):
            if not raw:
                return
            cleaned = self._clean_tag_raw(raw)
            if not cleaned:
                return
            tag = self._to_tag_form(cleaned)
            if not tag:
                return
            tag_norm = self._normalize(tag)
            if tag_norm in seen:
                return
            # basic filtering: no long sentences or punctuation
            if len(tag_norm) < 2 or len(tag_norm) > 60:
                return
            import string
            if any(ch in string.punctuation for ch in tag_norm):
                return
            seen.add(tag_norm)
            candidates.append(tag_norm)

        # Step 1: Core topic candidates (A)
        for base in list(core_bases):
            add_candidate(base)
            add_candidate(base + "정리")
            add_candidate(base + "정리")
            add_candidate(base + "정리방법")

        # Step 2: search-like noun-phrases (B)
        suffixes_b = ["정리방법", "정리팁", "정리노하우", "정리방법", "정리팁"]
        for base in list(core_bases):
            for sfx in suffixes_b:
                add_candidate(base + sfx)
        # season-aware combos
        seasons = [w for w in ["봄", "여름", "가을", "겨울", "계절"] if w in text_blob]
        for season in seasons:
            for base in list(core_bases):
                add_candidate(season + base + "정리")
                add_candidate(season + base)

        # Step 3: action/problem related (C)
        action_sfx = ["처리", "기부", "수거", "버리기", "판매"]
        for base in list(core_bases):
            for sfx in action_sfx:
                add_candidate(base + sfx)
        # commonly used problem phrases
        if "헌옷" in text_blob or "헌 옷" in text_blob:
            add_candidate("헌옷정리")
            add_candidate("헌옷처리")

        # Step 4: related concept combos (D)
        rel_sfx = ["재사용", "재활용", "중고", "순환"]
        for base in list(core_bases):
            for sfx in rel_sfx:
                add_candidate(base + sfx)

        # Step 5: useful concatenations from title/keyword n-grams (2-3)
        for src in (title_tokens, keyword_tokens, summary_tokens):
            for n in (2, 3):
                words = [w for w in src if len(w) > 1]
                for i in range(max(0, len(words) - n + 1)):
                    seq = "".join(words[i : i + n])
                    add_candidate(seq)

        # Ensure candidates are related: require at least one core token present
        filtered = []
        for c in candidates:
            # reject sentence-like strings
            if any(q in c for q in ["어떻게", "할까", "해야", "어디", "어떤"]):
                continue
            # require c to include at least one of core_bases or concept stems
            if not any(x in c for x in core_bases) and not any(s in c for s in present_stems):
                # allow if bigram of title/keyword present
                if not any(t in c for t in keyword_tokens + title_tokens):
                    continue
            filtered.append(c)

        # Cluster similar variants by root (strip common suffixes) and keep representative variants
        def root_of(tag: str) -> str:
            suf = ["방법", "팁", "노하우", "하기", "정보", "추천", "정리", "정리방법", "정리팁"]
            r = tag
            for s in suf:
                if r.endswith(s):
                    r = r[: -len(s)]
            return r if r else tag

        groups: Dict[str, List[str]] = {}
        for c in filtered:
            r = root_of(c)
            groups.setdefault(r, []).append(c)

        final_candidates: List[str] = []
        # for each group, choose up to 2 representatives (prefer shorter/base and one variant)
        for r, variants in groups.items():
            # sort variants by preference: shorter and containing '정리' or meaningful suffix
            variants_sorted = sorted(variants, key=lambda x: (len(x), '정리' not in x, x))
            pick = variants_sorted[:2]
            for p in pick:
                if p not in final_candidates:
                    final_candidates.append(p)
            if len(final_candidates) >= self.max_candidates:
                break

        # If still short, expand with more variants from groups
        if len(final_candidates) < min(target, self.max_candidates):
            for r, variants in groups.items():
                for v in variants:
                    if v not in final_candidates:
                        final_candidates.append(v)
                    if len(final_candidates) >= min(target, self.max_candidates):
                        break
                if len(final_candidates) >= min(target, self.max_candidates):
                    break

        return final_candidates[: min(target, self.max_candidates)]

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

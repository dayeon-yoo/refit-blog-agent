from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from models.schemas import BlogIdea, BlogPost


# Thresholds for simple token-based similarity (Jaccard)
TITLE_JACCARD_THRESHOLD = 0.5
ANGLE_JACCARD_THRESHOLD = 0.4


def _normalize(text: str) -> str:
    if not text:
        return ""
    # lower, remove punctuation, collapse whitespace
    t = text.lower()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokens(text: str) -> List[str]:
    return [t for t in _normalize(text).split(" ") if t]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = sa.intersection(sb)
    union = sa.union(sb)
    if not union:
        return 0.0
    return len(inter) / len(union)


def load_past_items(outputs_dir: Path = Path("outputs")) -> Dict[str, List[Dict]]:
    """Load past ideas and posts from outputs/latest_workflow.json and outputs/history/*.json"""
    past_ideas = []
    past_posts = []
    latest = outputs_dir / "latest_workflow.json"
    if latest.exists():
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            past_ideas.extend(data.get("ideas", []))
            past_posts.extend(data.get("blog_posts", []))
        except Exception:
            pass

    hist_dir = outputs_dir / "history"
    if hist_dir.exists() and hist_dir.is_dir():
        for p in hist_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                past_ideas.extend(data.get("ideas", []))
                past_posts.extend(data.get("blog_posts", []))
            except Exception:
                continue

    return {"ideas": past_ideas, "posts": past_posts}


def is_duplicate(idea: BlogIdea, past_ideas: List[Dict], past_posts: List[Dict]) -> bool:
    """Decide whether `idea` is a duplicate of any past idea/post.

    Rules implemented:
    - exact normalized title match -> duplicate
    - exact normalized keyword match + high angle overlap -> duplicate
    - title Jaccard similarity >= TITLE_JACCARD_THRESHOLD and angle Jaccard >= ANGLE_JACCARD_THRESHOLD -> duplicate
    - otherwise not duplicate
    """
    title = _normalize(idea.title)
    keyword = _normalize(idea.keyword)
    angle = _normalize(idea.angle)

    title_tokens = _tokens(title)
    angle_tokens = _tokens(angle)

    # check past ideas
    for p in past_ideas:
        p_title = _normalize(p.get("title", ""))
        p_keyword = _normalize(p.get("keyword", ""))
        p_angle = _normalize(p.get("angle", ""))

        # exact title match
        if title and p_title and title == p_title:
            return True

        # exact keyword match + angle similarity
        if keyword and p_keyword and keyword == p_keyword:
            # if either angle is empty, be conservative and do not mark duplicate
            if angle and p_angle:
                j = _jaccard(_tokens(angle), _tokens(p_angle))
                if j >= ANGLE_JACCARD_THRESHOLD:
                    return True

        # title similarity + angle similarity
        t_j = _jaccard(title_tokens, _tokens(p_title))
        a_j = _jaccard(angle_tokens, _tokens(p_angle))
        if t_j >= TITLE_JACCARD_THRESHOLD and a_j >= ANGLE_JACCARD_THRESHOLD:
            return True

    # also check past blog_posts (titles/summary) for title-level duplication
    for post in past_posts:
        p_title = _normalize(post.get("title", ""))
        if title and p_title and title == p_title:
            return True
        if _jaccard(title_tokens, _tokens(p_title)) >= TITLE_JACCARD_THRESHOLD:
            return True

    return False


def filter_duplicates(candidates: List[BlogIdea], outputs_dir: Path = Path("outputs")) -> List[BlogIdea]:
    loaded = load_past_items(outputs_dir)
    past_ideas = loaded.get("ideas", [])
    past_posts = loaded.get("posts", [])

    filtered = []
    for c in candidates:
        try:
            if not is_duplicate(c, past_ideas, past_posts):
                filtered.append(c)
        except Exception:
            # on error, keep candidate (fail-open)
            filtered.append(c)

    return filtered

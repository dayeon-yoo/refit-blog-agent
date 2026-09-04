from __future__ import annotations

import json

from config.settings import get_settings
from orchestrator.workflow import run_workflow


def main() -> None:
    settings = get_settings()
    if not settings.mock_mode:
        print("MOCK_MODE must be true for the current mock-only workflow.")
        return

    result = run_workflow()
    print(json.dumps({
        "ideas": [idea.model_dump() for idea in result.ideas[:3]],
        "top_3": [item.model_dump() for item in result.top_3],
        "blog_posts": [post.model_dump() for post in result.blog_posts],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

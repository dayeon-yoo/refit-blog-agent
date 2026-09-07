from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings
from orchestrator.workflow import run_workflow


def main() -> None:
    settings = get_settings()

    result = run_workflow()
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / "latest_workflow.json"
    result.save_json(latest_path, indent=2)

    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = history_dir / f"workflow_{timestamp}.json"
    result.save_json(history_path, indent=2)

    print(json.dumps({
        "ideas": [idea.model_dump() for idea in result.ideas[: settings.idea_count]],
        "top_3": [item.model_dump() for item in result.top_3[: settings.top_k]],
        "blog_posts": [post.model_dump() for post in result.blog_posts[: settings.top_k]],
        "saved_to": str(latest_path),
        "history_saved_to": str(history_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

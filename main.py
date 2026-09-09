from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings
from agents.content_planner_agent import ContentPlannerAgent
from agents.idea_agent import IdeaGenerator
from models.schemas import BlogIdea, IdeaCandidate, IdeaExpansionResult
from orchestrator.workflow import run_manual_workflow, run_workflow


LATEST_EXPANSION_PATH = Path(__file__).resolve().parent / "outputs" / "latest_expansion.json"


def manual_blog_idea() -> BlogIdea:
    """Editable sample input for the temporary manual workflow mode."""
    return BlogIdea(
        title="옷장에 안 입는 옷이 생기는 이유",
        keyword="안 입는 옷 정리",
        search_intent="정보 탐색",
        angle="안 입게 되는 옷의 특징을 살펴보고 재활용, 기부, 판매까지 연결",
        summary="옷장에 오래 남아 있는 옷을 점검하고 새로운 활용 방법을 소개한다.",
        rifit_connection="입지 않는 옷을 버리는 대신 다시 순환시킬 수 있는 방법을 안내한다.",
        seasonality=0.5,
    )


def refine_from_source(
    source: str,
    candidate_id: str,
    revision_request: str = "",
    generator: IdeaGenerator | None = None,
) -> BlogIdea:
    """Backward-compatible helper that expands once before selecting a candidate."""
    generator = generator or IdeaGenerator()
    expansion = generator.expand(source, candidate_count=3)

    try:
        candidate = select_candidate(expansion, candidate_id)
    except ValueError:
        available = ", ".join(item.candidate_id for item in expansion.candidates)
        raise ValueError(f"Unknown candidate_id {candidate_id}; available: {available}")
    return refine_selected_candidate(source, candidate, revision_request, generator=generator)


def _candidate_id_key(value: object) -> str:
    """Treat numeric IDs with or without the legacy prefix as equivalent."""
    value = str(value).strip()
    return value.removeprefix("candidate-")


def select_candidate(expansion: IdeaExpansionResult, candidate_id: str) -> IdeaCandidate:
    requested_id = _candidate_id_key(candidate_id)
    candidate = next(
        (
            item
            for item in expansion.candidates
            if _candidate_id_key(item.candidate_id) == requested_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"Candidate id {candidate_id} was not found in latest expansion.")
    return candidate


def refine_selected_candidate(
    source: str,
    candidate: IdeaCandidate,
    revision_request: str = "",
    generator: IdeaGenerator | None = None,
) -> BlogIdea:
    """Refine an already selected candidate without expanding again."""
    generator = generator or IdeaGenerator()
    return generator.refine(source, candidate, revision_request)


def save_latest_expansion(expansion: IdeaExpansionResult, path: Path = LATEST_EXPANSION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expansion.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_latest_expansion(source: str, path: Path = LATEST_EXPANSION_PATH) -> IdeaExpansionResult:
    if not path.exists():
        raise ValueError("Latest expansion was not found. Run --expand again for this source.")
    expansion = IdeaExpansionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if expansion.source_input != source:
        raise ValueError(
            "Latest expansion source does not match --source. "
            "Run --expand again for this source."
        )
    return expansion


def run_refined_manual_workflow(
    source: str,
    candidate_id: str | None = None,
    revision_request: str = "",
    generator: IdeaGenerator | None = None,
    writer=None,
    tag_agent=None,
    image_agent=None,
    content_planner=None,
    candidate: IdeaCandidate | None = None,
):
    """Refine one selected candidate, then reuse the manual pipeline once."""
    if candidate is None:
        if candidate_id is None:
            raise ValueError("candidate_id or candidate is required")
        idea = refine_from_source(source, candidate_id, revision_request, generator=generator)
    else:
        idea = refine_selected_candidate(source, candidate, revision_request, generator=generator)
    planner = content_planner or ContentPlannerAgent()
    content_plan = planner.plan(source, idea)
    result = run_manual_workflow(
        idea,
        writer=writer,
        tag_agent=tag_agent,
        image_agent=image_agent,
        source=source,
        content_plan=content_plan,
    )
    return idea, content_plan, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RIFIT blog workflow")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Run the Writer -> Tag -> Image pipeline with manual_blog_idea()",
    )
    parser.add_argument(
        "--expand",
        metavar="IDEA",
        help="Expand one user idea into content candidates without running the workflow",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Load the latest expansion, select a candidate, and print the refined BlogIdea",
    )
    parser.add_argument(
        "--full-manual",
        action="store_true",
        help="Load one candidate, refine it, then run the Writer -> Tag -> Image pipeline",
    )
    parser.add_argument("--source", help="Source input used with --refine")
    parser.add_argument("--candidate-id", help="Candidate number or id used with --refine")
    parser.add_argument("--revision", default="", help="Optional refinement request")
    args = parser.parse_args()
    settings = get_settings()

    if args.full_manual:
        if not args.source or not args.candidate_id:
            parser.error("--full-manual requires --source and --candidate-id")
        expansion = load_latest_expansion(args.source)
        candidate = select_candidate(expansion, args.candidate_id)
        idea, content_plan, result = run_refined_manual_workflow(
            args.source,
            revision_request=args.revision,
            candidate=candidate,
        )
        print(json.dumps({
            "refined_blog_idea": idea.model_dump(mode="json"),
            "content_plan": content_plan.model_dump(mode="json"),
            "workflow": result.model_dump(mode="json"),
        }, ensure_ascii=False, indent=2))
        return

    if args.expand:
        result = IdeaGenerator().expand(args.expand)
        save_latest_expansion(result)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.refine:
        if not args.source or not args.candidate_id:
            parser.error("--refine requires --source and --candidate-id")
        expansion = load_latest_expansion(args.source)
        candidate = select_candidate(expansion, args.candidate_id)
        idea = refine_selected_candidate(args.source, candidate, args.revision)
        print(json.dumps(idea.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    result = run_manual_workflow(manual_blog_idea()) if args.manual else run_workflow()
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

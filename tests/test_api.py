from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import get_blog_service
from config.settings import get_settings
from llm.client import OpenAILLMClient
from models.schemas import BlogIdea
from services.blog_workflow_service import BlogWorkflowService

SOURCE = "민트코어가 유행 끝나가는 것 같고 레몬코어나 포도색이 새롭게 뜨는 것 같아."

FREE_SOURCES = [
    "체크셔츠를 사려다 빈티지로 사기로 마음먹고 오슬로에서 구입했어요. "
    "옷을 새로 구매하는데에는 빈티지도 하나의 방법이라는 걸 깨달았습니다.",
    "니트나 맨투맨에 보풀이 생기는 이유와 집에서 관리할 수 있는 방법을 순서대로 정리하고 싶다.",
    "도자기에 생긴 균열을 살펴보고 관리할 때 주의할 점을 적고 싶습니다.",
]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("KEYWORD_PROVIDER", "mock")
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(OpenAILLMClient, "generate_structured", Mock(side_effect=AssertionError("Real API forbidden")))
    yield create_app()
    get_settings.cache_clear()


def test_health_does_not_invoke_service(app):
    app.dependency_overrides[get_blog_service] = Mock(side_effect=AssertionError("No service needed"))
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-RIFIT-LLM-Mode"] == "mock"


def test_mock_mode_header_is_readable_by_frontend(app):
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert "X-RIFIT-LLM-Mode" in response.headers["access-control-expose-headers"]


def test_explicit_live_expansion_uses_real_client_without_switching_server_mode(app, monkeypatch):
    from llm.client import MockLLMClient
    import services.blog_workflow_service as service_module

    fake = MockLLMClient()
    fake.client = Mock()
    constructor = Mock(return_value=fake)
    monkeypatch.setattr(service_module, "OpenAILLMClient", constructor)
    with TestClient(app) as client:
        response = client.post("/api/ideas/expand", json={"source": SOURCE, "use_live": True})
    assert response.status_code == 200, response.text
    assert response.headers["X-RIFIT-LLM-Mode"] == "live"
    constructor.assert_called_once()
    assert get_settings().mock_mode is True
    assert len(fake.calls) == 1


def test_default_mock_expansion_never_constructs_live_client(app, monkeypatch):
    import services.blog_workflow_service as service_module
    constructor = Mock(side_effect=AssertionError("No paid opt-in"))
    monkeypatch.setattr(service_module, "OpenAILLMClient", constructor)
    with TestClient(app) as client:
        response = client.post("/api/ideas/expand", json={"source": SOURCE})
    assert response.status_code == 200
    constructor.assert_not_called()


@pytest.mark.parametrize("source", FREE_SOURCES)
def test_free_source_expansion_mock_is_deterministic_and_contract_valid(app, source):
    from agents.idea_agent import IdeaGenerator
    from models.schemas import IdeaExpansionResult

    with TestClient(app) as client:
        response = client.post("/api/ideas/expand", json={"source": source})
        repeated = client.post("/api/ideas/expand", json={"source": source})
    assert response.status_code == 200, response.text
    assert repeated.json() == response.json()
    result = IdeaExpansionResult.model_validate(response.json())
    assert set(response.json()) == {"source_input", "candidates"}
    assert result.source_input == source
    assert all(source not in candidate.title for candidate in result.candidates)
    IdeaGenerator._validate_expansion(source, result, 3)


@pytest.mark.parametrize("count", [3, 5])
def test_mock_fallback_works_for_both_expansion_schemas_with_raw_evidence(count):
    from agents.idea_agent import IdeaGenerator
    from llm.client import MockLLMClient
    from models.schemas import IdeaExpansionResult, InternalIdeaExpansion

    source = FREE_SOURCES[0]
    client = MockLLMClient()
    payload = {"user_input": source, "candidate_count": count}
    internal = client.generate_structured("mock", InternalIdeaExpansion, payload)
    public = client.generate_structured("mock", IdeaExpansionResult, payload)
    assert internal.candidates == public.candidates
    assert IdeaGenerator._core_source_intent_provenance_issues(source, internal.source_intent) == []
    IdeaGenerator._validate_expansion(source, public, count)
    normal_client = MockLLMClient()
    IdeaGenerator(llm_client=normal_client).expand(source, candidate_count=count)
    assert len(normal_client.calls) == 1


def test_mock_api_pipeline_uses_request_candidate_without_persistence(app, monkeypatch, tmp_path):
    import main
    from agents.content_planner_agent import ContentPlannerAgent
    from agents.idea_agent import IdeaGenerator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "load_latest_expansion", Mock(side_effect=AssertionError("No CLI persistence")))
    original_plan = ContentPlannerAgent.plan
    plans = []

    def plan(self, source, idea):
        plans.append((source, idea))
        return original_plan(self, source, idea)

    monkeypatch.setattr(ContentPlannerAgent, "plan", plan)
    with TestClient(app) as client:
        expanded = client.post("/api/ideas/expand", json={"source": SOURCE})
        assert expanded.status_code == 200
        assert set(expanded.json()) == {"source_input", "candidates"}
        candidate = expanded.json()["candidates"][1]
        monkeypatch.setattr(IdeaGenerator, "expand", Mock(side_effect=AssertionError("Must not expand again")))
        original_refine = IdeaGenerator.refine
        received = []

        def refine(self, source, selected, revision):
            received.append((source, selected.model_dump(), revision))
            return original_refine(self, source, selected, revision)

        monkeypatch.setattr(IdeaGenerator, "refine", refine)
        refined = client.post("/api/ideas/refine", json={
            "source": SOURCE, "candidate": candidate, "revision_request": "",
        })
        assert refined.status_code == 200, refined.text
        assert received == [(SOURCE, candidate, "")]
        generated = client.post("/api/workflow/generate", json={"source": SOURCE, "blog_idea": refined.json()})
        assert generated.status_code == 200, generated.text
        data = generated.json()
        assert data["blog_idea"] == refined.json()
        assert data["content_plan"]["writing_script"]
        assert data["post"]["content"]
        assert data["tags"] == data["post"]["tags"]
        assert data["image_plan"] == data["post"]["image_plan"]
        assert data["qc"]["status"] in {"PASS", "NEEDS_REVISION", "BLOCK"}
        assert len(plans) == 1
        assert plans[0] == (SOURCE, BlogIdea.model_validate(refined.json()))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("path,payload", [
    ("/api/ideas/expand", {"source": "  "}),
    ("/api/ideas/refine", {"source": SOURCE}),
    ("/api/workflow/generate", {"source": SOURCE, "blog_idea": {}}),
])
def test_request_validation_rejects_invalid_input(app, path, payload):
    app.dependency_overrides[get_blog_service] = lambda: Mock(spec=BlogWorkflowService)
    with TestClient(app) as client:
        assert client.post(path, json=payload).status_code == 422


@pytest.mark.parametrize("error,status,detail", [
    (ValueError("Idea candidate is unrelated to the source input"), 422, "Idea candidate is unrelated to the source input"),
    (RuntimeError("private internal data"), 500, "Internal server error"),
])
def test_errors_return_http_details_without_traceback(app, error, status, detail):
    service = Mock(spec=BlogWorkflowService)
    service.expand.side_effect = error
    app.dependency_overrides[get_blog_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/ideas/expand", json={"source": SOURCE})
    assert response.status_code == status
    assert response.json() == {"detail": detail}


def test_cors_allows_only_configured_origins(app, monkeypatch):
    with TestClient(app) as client:
        for origin, allowed in [("http://localhost:5173", True), ("https://untrusted.example", False)]:
            response = client.options("/api/ideas/expand", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            })
            assert (response.headers.get("access-control-allow-origin") == origin) is allowed
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in response.headers
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://frontend.example")
    with TestClient(create_app()) as client:
        response = client.get("/api/health", headers={"Origin": "https://frontend.example"})
        assert response.headers["access-control-allow-origin"] == "https://frontend.example"

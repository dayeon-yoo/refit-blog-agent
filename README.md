# RIFIT Blog Agent

A mock-first blog workflow for generating sustainable fashion article concepts and draft content for a RIFIT-style brand.

## Goal

This project is designed to validate the structure and data flow of a multi-agent blog generation pipeline without making real external API calls.

The current focus is:

- Mock mode only
- Agent-to-agent data continuity
- Prompt and schema quality before real LLM integration
- Stable workflow execution and test validation

## Architecture

Trend → SEO → Idea → Brand → Critic → TOP3 → Writer

### Agent responsibilities

- TrendAgent
  - identifies blog-worthy trend themes
  - returns structured trend candidates

- SEOAgent
  - converts trend themes into keyword-focused SEO candidates
  - includes search intent, seasonality, and content potential

- IdeaGenerator
  - combines trend and SEO context into blog topic candidates
  - generates 10+ ideas in mock mode

- BrandAgent
  - evaluates each idea against brand fit and content guidelines
  - assigns fit_score and brand_alignment

- CriticAgent
  - scores ideas based on real content quality and brand alignment
  - selects the top 3 ideas

- WriterAgent
  - writes final blog post content using the selected idea, critic score, and brand context

## Main files

- [main.py](main.py): local execution entry point for mock workflow
- [orchestrator/workflow.py](orchestrator/workflow.py): end-to-end agent orchestration
- [agents](agents): individual agents
- [models/schemas.py](models/schemas.py): typed request/response schemas
- [config/settings.py](config/settings.py): environment-based configuration
- [prompts](prompts): prompt templates for each stage
- [tests/test_workflow.py](tests/test_workflow.py): mock workflow verification

## Mock-only execution

Set the environment to mock mode:

```bash
export MOCK_MODE=true
```

Then run:

```bash
python main.py
```

or:

```bash
python -m pytest -q
```

To run one manually provided BlogIdea through the existing Writer, Tag, and
Image stages without invoking the Idea Agent, edit `manual_blog_idea()` in
`main.py` and run:

```bash
python main.py --manual
```

The result is saved to `outputs/latest_workflow.json` using the same output
path as the automatic workflow.

## Naver Search Ads keyword provider

The default provider remains the deterministic mock provider. To use the Naver
Search Ads keyword tool explicitly, set `MOCK_MODE=false` and
`KEYWORD_PROVIDER=naver`, then provide these values in `.env`:

```dotenv
KEYWORD_PROVIDER=naver
NAVER_SEARCHAD_API_KEY=
NAVER_SEARCHAD_SECRET_KEY=
NAVER_SEARCHAD_CUSTOMER_ID=
```

To issue credentials, sign in to the Naver Search Ads advertiser center, open
`Tools > API Manager`, accept the API terms, and create an API license. Use the
issued API key and secret key, and copy the advertiser Customer ID into `.env`.
Never commit real credentials.

## Important constraints

This project intentionally does not perform any of the following in the current phase:

- OpenAI API calls
- real web search
- Naver API
- vector DB
- DB or scheduler integrations
- publisher or analytics integrations

The focus is on code quality, schema stability, and workflow correctness before API integration.

## Validation

The workflow is expected to maintain the following checks in mock mode:

- Trend results are created
- SEO results are tied to trend context
- 10+ ideas are generated
- each idea has keyword / search intent / angle
- brand evaluation exists for every idea
- Critic selects exactly 3 ideas
- Writer produces blog posts for each top idea
- blog content is not empty

## Local HTTP API

Install dependencies with `python3 -m pip install -r requirements.txt`, then run:

```bash
MOCK_MODE=true KEYWORD_PROVIDER=mock uvicorn api.app:app --reload
```

The API is stateless. The frontend keeps the source, selected candidate, and
refined idea between requests. It does not read or write `outputs/latest_expansion.json`.

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /api/health` | None | `{"status":"ok"}` |
| `POST /api/ideas/expand` | `{"source":"..."}` | Existing `IdeaExpansionResult` |
| `POST /api/ideas/refine` | `{"source":"...","candidate":{...},"revision_request":""}` | Existing `BlogIdea` |
| `POST /api/workflow/generate` | `{"source":"...","blog_idea":{...}}` | `blog_idea`, `content_plan`, `post`, `tags`, `image_plan`, `qc` |

Interactive request schemas are available at `/docs`. Refinement uses the supplied
candidate directly without expanding again. Generation plans once and delegates
Writer, Tag, Image Plan, and QC to the existing manual workflow. A QC `BLOCK` is
returned as a completed review in the response; generation validation failures
return HTTP 422. Unexpected failures return a generic HTTP 500 without traceback.

Local development permits `http://localhost:5173`. Set `CORS_ALLOW_ORIGINS` to a
comma-separated list of explicit origins to override it. With `APP_ENV=production`,
no origins are allowed by default. Wildcard origins are rejected.

Run API and CLI regression tests without external API calls:

```bash
MOCK_MODE=true KEYWORD_PROVIDER=mock python3 -B -m pytest -q
```

## Web Workspace

RIFIT logo: `frontend/public/refit-logo.png`.
Place the actual logo image at that path; the header shows a text fallback when
the image is unavailable. No logo asset is generated by the application.

Start the backend in mock mode:

```bash
MOCK_MODE=true KEYWORD_PROVIDER=mock uvicorn api.app:app --reload
```

Mock mode is only for testing the UI and API connection. Its fallback candidates
are fixed templates, not AI-generated editorial proposals. The UI labels these
responses as samples; repeating expansion in mock mode will not improve them.

For real idea development, stop the mock backend and restart it explicitly:

```bash
MOCK_MODE=false KEYWORD_PROVIDER=mock uvicorn api.app:app --reload
```

This requires `OPENAI_API_KEY` in the backend environment and incurs OpenAI costs.
Keep `KEYWORD_PROVIDER=mock` to avoid Naver requests. Expanding an idea makes only
the existing Expansion call (plus at most one validation correction); refinement
and article generation run only when their respective buttons are clicked.
An exported `MOCK_MODE=true` or the launch command overrides the `.env` value.
No API mode is switched automatically by the UI.

Step 2 also has an explicit **실제 AI로 후보 3개 생성** button. It sends
`use_live: true` only to `/api/ideas/expand`, even when the server is in mock mode.
This requires a configured OpenAI key and is a paid opt-in, not an automatic
retry. The server disables SDK network retries for this path; the existing one
validation-correction budget remains (at most two LLM calls). It does not run
refinement or article generation, or change their server mode. Existing candidates
stay visible if the request fails. The public expansion response is unchanged.

Candidate cards show `title`, the reader-facing subtitle in `perspective`, and
the one-to-two-sentence direction in `brief_description`. The internal editorial
question remains available to refinement but is not displayed as the subtitle.

In another terminal (Node.js 22+ recommended):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The workspace keeps your source, candidate selection,
revision request, and results in memory until a page refresh or a new idea.
Choose a candidate, confirm its direction, then generate and review the draft.
QC revision requests do not hide the draft. Image cards describe a plan, not
generated images. Copy buttons require browser clipboard permission.

The frontend defaults to http://localhost:8000. To change it, set
`VITE_API_BASE_URL` in `frontend/.env.local` (see `frontend/.env.example`) and
restart Vite. The backend CORS origin must match the frontend origin.

Frontend verification: `npm test` and `npm run build` inside `frontend/`.
Tests use stubbed responses and make no backend or external API requests.

## Note

Real LLM integration can be added later behind the same interface once the codebase, prompts, and validation logic are mature enough.

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

## Note

Real LLM integration can be added later behind the same interface once the codebase, prompts, and validation logic are mature enough.

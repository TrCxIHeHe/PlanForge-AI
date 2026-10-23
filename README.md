# Autonomous Document Agent

A FastAPI service with one endpoint, `POST /agent`. Give it a plain-English request and it:

1. **Plans**: turns the request into explicit *requirements* and a *task graph* (its own TODO list).
2. **Executes**: runs tasks as a dependency graph, calling **tools** for exact numbers and dates. Concurrency is configurable and defaults to one for local hardware.
3. **Reflects**: checks the draft against its own requirements and rewrites *only* the sections that fail.
4. **Delivers**: a polished Word (`.docx`) file plus a JSON trace of everything the agent decided.

Runs on a local **Ollama** model by default, or an OpenAI-compatible endpoint such as **Gemini**, **Groq**, or LM Studio. A cloud provider can use Ollama as a fallback; never commit API keys.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# The default local model is llama3.2:3b. Set OLLAMA_MODEL to another installed Ollama model if desired.
# For Gemini, configure the local .env values below, then run: python scripts/check_gemini.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/docs>, or:

```bash
python scripts/run_examples.py all        # runs both test inputs, prints the task list, saves output/demo/*.docx
```

Tests (no model needed, ~2 s): `pytest -q`. No model at all? `python scripts/offline_demo.py` runs the full pipeline with a scripted fake LLM and writes a sample `.docx` (canned prose, real pipeline).

For Gemini, set `LLM_PROVIDER=openai`, `LLM_FALLBACK_PROVIDER=ollama`, `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`, `OPENAI_MODEL`, and `GEMINI_API_KEY` in the local `.env` file. Run `python scripts/check_gemini.py` to verify authenticated model access; it selects an account-accessible `generateContent` model if the configured one is unavailable and never prints the key. Gemini model and endpoint are configurable; Ollama remains the automatic fallback. Other OpenAI-compatible providers continue to use `OPENAI_API_KEY`.

## API

`POST /agent` with `{"request": "..."}` returns:

| field | meaning |
|---|---|
| `summary`, `status` | `completed` or `completed_with_warnings`, plus a plain-English summary |
| `plan` | title, assumptions, **requirements**, **tasks** (dependencies, requirement ids, tools), plan source and any repairs |
| `execution` | per-task status, attempts, tools used |
| `quality_check` | per-requirement verdict (`met` / `not_met` / `unverified`), revision rounds and revised tasks |
| `warnings` | everything a reader should double-check (failed sections, unmet requirements, tool failures, fallback used) |
| `llm` | provider, model, calls, retries, JSON repairs, fallbacks, latency (per request) |
| `document` | filename, `download_path` (`GET /documents/{file}`), size, base64 |

Errors: `422` invalid/rejected request, `503` model unavailable (clear message), `502` agent failure, `500` unexpected (no stack trace leaked). `GET /health` reports the active provider and tools.

## Architecture

```
POST /agent -> validate -> PLAN -> EXECUTE (task DAG + tools) -> REVIEW -> targeted REVISE -> .docx
                            |            |                          |
                      requirements   parallel waves,            per-requirement
                      + task graph   retries, failure           verdicts, rewrite only
                      (repair loop,  isolation                  failing sections
                      fallback plan)
```

| module | responsibility |
|---|---|
| `llm.py` | Provider clients (Ollama, OpenAI-compatible), retry with backoff, JSON repair, optional provider fallback, per-request usage stats |
| `planner.py` | LLM plan -> **normalised, validated DAG**: renumbers ids, drops unknown tools/deps, breaks cycles, guarantees every requirement is owned by a task; one self-repair round; deterministic fallback plan |
| `tools.py` | `calculate` (AST-based safe evaluator, no `eval`), `schedule` (real calendar dates), `risk_score` (ranked register). Tools never raise |
| `executor.py` | Runs the DAG with `asyncio` + a concurrency limit; per-task retries; a failed task becomes a marked placeholder and never aborts siblings |
| `reviewer.py` | Reflection: LLM verdict per requirement + deterministic checks (failed sections, placeholder text) -> targeted feedback |
| `blocks.py` | Parses/normalises model output into paragraph / list / table blocks, strips markdown noise |
| `document.py` | A4 Word output: styled tables with repeating headers, page X of Y, assumptions, review notes, appendices (plan, traceability, tool calls) |
| `agent.py` | Orchestration, input guardrails, warnings, summary |

## The one engineering improvement: reflection against traceable requirements

**What.** The planner first writes a numbered list of testable *requirements* from the request (including implicit ones) and assigns each to the task(s) that must satisfy it. After drafting, a reviewer checks every requirement against the draft. Unmet requirements map back, through the traceability links, to the exact sections responsible, and *only those* are rewritten with the reviewer's feedback. The loop is bounded (`MAX_REVISION_ROUNDS`).

**Why.** Small free models often drift from the request: a missing budget, no comparison, no risks. A generic "improve this" pass is vague and costly. Tying review to explicit requirements makes it targeted, cheap and auditable, and the Word appendix shows exactly which task covered which requirement.

**How it improves the agent.** Missing content is caught and fixed without a human. Whatever can't be fixed is reported honestly (`not_met`, warnings, and a review-notes block in the document) instead of being hidden. If the reviewer itself is down, the run still delivers and marks coverage `unverified`.

Supporting design (not separate features, they make reflection and planning trustworthy): tool calling for exact arithmetic/dates; validated plans with a repair loop and fallback; retries, JSON repair and provider fallback; failure isolation.

## Design decisions and trade-offs

- **Autonomous planning vs deterministic workflow.** The model decides the sections, dependencies and tools, but everything it returns is validated and repaired, and a fixed fallback plan exists. Freedom where it adds value, guarantees where it matters.
- **Plan-declared tools plus on-demand tools.** The planner attaches tool calls to tasks; a writer may also request tools mid-task. Results are authoritative in the prompt, so the model never does arithmetic itself.
- **Single agent, many steps.** One planner/writer/reviewer role set over one model is simpler to debug than multi-agent. The DAG executor leaves room to swap in specialised workers or a queue later.
- **Stateless service.** A run holds no shared state (per-request stats use a `ContextVar`), so it scales horizontally; moving long runs to a job queue is the obvious next step.

## Limits (honest)

- The default local configuration uses one concurrent task, a 180-second model timeout, and two LLM attempts to limit resource pressure on CPU-only laptops. Real-model quality depends on the model you run; verify both example inputs against your model before recording.
- Review is another model call: it can be wrong, so `met` means "the reviewer believes so", and warnings are surfaced.
- The agent writes from model knowledge plus mock/illustrative assumptions (clearly labelled); it has no web or document retrieval.
- Prompts fence user text in `<request>` tags (typed tags are neutralised). That is a basic prompt boundary, not a robust defence against prompt injection.
- Deterministic checks (failed sections, leftover placeholder text) always override a positive LLM review: the status becomes `needs_attention` and a warning is added.
- If the reviewer omits any requirement verdict, the overall quality status remains `unverified`; partial review coverage cannot produce a `passed` result.
- Long runs are synchronous HTTP; with a slow local model use a generous client timeout (the example script waits up to 30 min).

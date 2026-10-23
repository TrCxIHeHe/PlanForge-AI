# Video script (8-10 min)

**Before recording:** keep the Gemini API key only in the local `.env` file. Run `python scripts/check_gemini.py` to verify account access to the configured text model; if it selects another supported model, restart the API. Ollama is configured as the automatic fallback. Run `python scripts/run_examples.py all` and inspect warnings before recording; do not describe a fallback run as Gemini output. Keep `output/demo/*.docx` ready to open and logs visible.

## 1. Live demo (3-4 min)
1. `GET /health` in `/docs`: show provider/model and the three tools.
2. **Standard request** (`examples/request_standard.json`): call `POST /agent`. Point at the response: the requirement list, the **task list with dependencies**, tools used, quality check. Open the `.docx`: tables, page numbers, Appendix A (plan) and B (traceability).
3. **Complex request** (`examples/request_complex.json`): ambiguous + conflicting (nothing budget vs 24/7). Show the **assumptions** it invented, how it resolved the conflict, real dates from the `schedule` tool, ranked risks from `risk_score`, and any revision it triggered.

## 2. What I built (2-3 min)
Flow: validate -> plan -> DAG execute with tools -> reflect -> targeted revise -> docx. Walk the module table in the README. Stress: the model *proposes* a plan, my code *validates and repairs* it (cycles, unknown tools, uncovered requirements); the DAG honors dependencies and concurrency defaults to 1 on this laptop; exec summary waits for its dependencies.

**The one improvement: reflection against traceable requirements.** Why: small models drift; requirement-linked review is targeted, cheap and auditable. Show a revision in the trace (`revised_tasks`). Mention supporting pieces briefly: tool calling for exact numbers, retries/fallback.

## 3. Debugging insight (1-2 min)
**Issue:** a task that returned unusable output got only one try even though `TASK_MAX_ATTEMPTS=2`, and the section was marked failed.
**Root cause:** the attempt counter was incremented once when the call returned and again in the `ContentError` handler, so the budget was spent after one failure.
**Fix:** count attempts in exactly one place; added `test_task_retry_recovers_from_unusable_output`, which fails on the old code. Lesson: retry logic needs tests that force the first attempt to fail.

## 4. Trade-off (1-2 min)
**Autonomous planning vs deterministic workflow.** Fully autonomous = flexible but unpredictable with small models; fully fixed = reliable but generic. I let the model plan, then constrain it: validate, repair, one self-correction round, deterministic fallback. Cost: more code than a prompt-and-pray agent. Gain: it degrades gracefully, and the output says when it did. Extension: job queue for long runs, specialised workers per section type, retrieval for real facts.

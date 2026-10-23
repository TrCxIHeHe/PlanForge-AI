"""Run the full agent pipeline with a scripted fake LLM (no model needed) and write a sample .docx.

This exercises planning, the task DAG, tools, review and Word generation. The prose is canned, so use
scripts/run_examples.py against a running server for real model output.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import DocumentAgent  # noqa: E402
from app.config import Settings  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402


async def main() -> None:
    out = Path(__file__).resolve().parents[1] / "output" / "documents"
    agent = DocumentAgent(Settings(output_dir=out), llm=FakeLLM())
    resp = await agent.run("Prepare a proposal for reducing support response time at a growing SaaS business within 90 days.")
    print(resp.summary)
    for step in resp.execution:
        print(f"  {step.task_id} {step.status:9} deps={step.depends_on} tools={step.tools_used}")
    print("document:", out / resp.document.filename)


if __name__ == "__main__":
    asyncio.run(main())

"""FastAPI application entry point."""
import logging
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .agent import AgentExecutionError, DocumentAgent, RequestRejected
from .config import Settings
from .llm import LLMError
from .schemas import DOCX_MIME, AgentRequest, AgentResponse

log = logging.getLogger("agent.api")


def create_app(agent: DocumentAgent | None = None, settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per LLM call is noise
    doc_agent = agent or DocumentAgent(cfg)

    app = FastAPI(
        title=cfg.app_name,
        version="2.0.0",
        description="Autonomous agent: plans its own task graph, runs tools, self-reviews against requirements and returns a Word document.",
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": cfg.app_name, "status": "ready", "docs": "/docs", "agent_endpoint": "POST /agent"}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "llm": doc_agent.llm.describe(), "tools": doc_agent.tools.names, "output_directory_ready": cfg.output_dir.exists()}

    @app.post("/agent", response_model=AgentResponse)
    async def run_agent(payload: AgentRequest) -> AgentResponse:
        try:
            return await doc_agent.run(payload.request)
        except RequestRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=f"The language model is unavailable: {exc}") from exc
        except AgentExecutionError as exc:
            raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc
        except Exception as exc:  # never leak a stack trace to the client
            log.exception("unhandled error in /agent")
            raise HTTPException(status_code=500, detail="Internal error while running the agent.") from exc

    @app.get("/documents/{filename}")
    async def download_document(filename: str) -> FileResponse:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+\.docx", filename):  # block traversal and arbitrary file access
            raise HTTPException(status_code=400, detail="Invalid document filename.")
        path = (cfg.output_dir / filename).resolve()
        if path.parent != cfg.output_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="Document not found.")
        return FileResponse(path=path, media_type=DOCX_MIME, filename=filename)

    return app


app = create_app()

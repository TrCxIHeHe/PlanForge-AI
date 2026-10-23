import httpx
from fastapi.testclient import TestClient

from app.agent import DocumentAgent
from app.llm import LLMError
from app.main import create_app
from tests.fakes import FakeLLM, make_settings

REQ = {"request": "Prepare a proposal for reducing support response time at a growing SaaS business."}


def client(tmp_path, llm=None, **kw):
    cfg = make_settings(tmp_path, **kw)
    return TestClient(create_app(agent=DocumentAgent(cfg, llm=llm or FakeLLM()), settings=cfg))


def test_post_agent_returns_plan_execution_and_document(tmp_path):
    c = client(tmp_path)
    body = c.post("/agent", json=REQ).json()
    assert body["status"] == "completed" and len(body["plan"]["tasks"]) == 4 and body["execution"][0]["task_id"] == "T1"
    dl = c.get(body["document"]["download_path"])
    assert dl.status_code == 200 and dl.content[:2] == b"PK" and "wordprocessingml" in dl.headers["content-type"]


def test_validation_errors_are_422(tmp_path):
    c = client(tmp_path)
    assert c.post("/agent", json={}).status_code == 422
    assert c.post("/agent", json={"request": "short"}).status_code == 422
    r = c.post("/agent", json={"request": "?!?! 1234 ???? 5678 ####"})
    assert r.status_code == 422 and "natural-language" in r.json()["detail"]


def test_llm_outage_is_503_not_a_stack_trace(tmp_path):
    r = client(tmp_path, FakeLLM({"plan": LLMError("ollama down")})).post("/agent", json=REQ)
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_unexpected_error_is_a_clean_500(tmp_path):
    r = client(tmp_path, FakeLLM({"plan": RuntimeError("bug")})).post("/agent", json=REQ)
    assert r.status_code == 500 and "bug" not in r.text


def test_health_and_download_guards(tmp_path):
    c = client(tmp_path)
    h = c.get("/health").json()
    assert h["status"] == "ok" and "calculate" in h["tools"]
    assert c.get("/documents/missing.docx").status_code == 404
    assert c.get("/documents/not-a-docx.txt").status_code == 400

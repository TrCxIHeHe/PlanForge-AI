import httpx
import pytest

from app.config import Settings
from app.llm import FallbackLLM, LLMError, LLMStats, OllamaClient, OpenAICompatClient, build_llm, extract_json, request_stats


def ollama_reply(content):
    return httpx.Response(200, json={"message": {"content": content}})


async def no_sleep(_):
    return None


def client(handler, **kw):
    return OllamaClient(Settings(llm_max_retries=3, llm_backoff_seconds=0), transport=httpx.MockTransport(handler), sleep=no_sleep, **kw)


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} hope it helps', '{"a": 1,}'])
def test_extract_json_tolerates_common_model_noise(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]"])
def test_extract_json_rejects_garbage(text):
    with pytest.raises(ValueError):
        extract_json(text)


async def test_retries_transient_http_errors_then_succeeds():
    seen = []

    def handler(req):
        seen.append(1)
        return httpx.Response(503) if len(seen) < 3 else ollama_reply('{"ok": true}')

    c = client(handler)
    assert await c.generate_json(system="s", user="u") == {"ok": True}
    assert len(seen) == 3 and c.stats.retries == 2


async def test_invalid_json_triggers_self_repair_prompt():
    bodies = []

    def handler(req):
        bodies.append(req.content.decode())
        return ollama_reply("not json at all") if len(bodies) == 1 else ollama_reply('{"fixed": 1}')

    c = client(handler)
    assert await c.generate_json(system="s", user="u") == {"fixed": 1}
    assert c.stats.json_repairs == 1 and "not a single valid JSON object" in bodies[1]


async def test_auth_and_missing_model_fail_fast_without_retry():
    seen = []

    def handler(req):
        seen.append(1)
        return httpx.Response(404)

    with pytest.raises(LLMError, match="not found"):
        await client(handler).generate_json(system="s", user="u")
    assert len(seen) == 1


async def test_gives_up_after_max_retries_with_clear_error():
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        await client(lambda req: httpx.Response(500)).generate_json(system="s", user="u")


async def test_connection_error_is_retried_then_reported():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="ConnectError"):
        await client(handler).generate_json(system="s", user="u")


async def test_openai_compat_requires_key_and_parses_choices():
    s = Settings(openai_api_key="", llm_backoff_seconds=0)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        await OpenAICompatClient(s, transport=httpx.MockTransport(lambda r: httpx.Response(200))).generate_json(system="s", user="u")
    s = Settings(openai_api_key="k", llm_backoff_seconds=0)
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"x": 2}'}}]})

    ok = httpx.MockTransport(handler)
    assert await OpenAICompatClient(s, transport=ok).generate_json(system="s", user="u") == {"x": 2}
    assert seen["authorization"] == "Bearer k"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"


async def test_gemini_requires_gemini_api_key_and_redacts_keys_from_errors():
    settings = Settings(
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openai_api_key="must-not-be-used-for-gemini",
        gemini_api_key="secret-test-key",
        llm_max_retries=1,
    )

    def handler(_request):
        raise httpx.ConnectError("connection failed secret-test-key")

    with pytest.raises(LLMError) as error:
        await OpenAICompatClient(
            settings,
            transport=httpx.MockTransport(handler),
            sleep=no_sleep,
        ).generate_json(system="s", user="u")
    assert "secret-test-key" not in str(error.value)
    assert "must-not-be-used-for-gemini" not in str(error.value)

    missing = Settings(
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openai_api_key="legacy-openai-key",
        gemini_api_key="",
        llm_max_retries=1,
    )
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        await OpenAICompatClient(missing).generate_json(system="s", user="u")


async def test_gemini_openai_compatible_endpoint_and_ollama_fallback_configuration():
    s = Settings(
        llm_provider="openai",
        llm_fallback_provider="ollama",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openai_model="gemini-3.8-flash",
        gemini_api_key="test-key",
        llm_max_retries=1,
        llm_backoff_seconds=0,
    )
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("Authorization")))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"provider":"gemini"}'}}]})

    llm = build_llm(s, transport=httpx.MockTransport(handler))
    assert await llm.generate_json(system="s", user="u") == {"provider": "gemini"}
    assert seen == [(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "Bearer test-key",
    )]


async def test_fallback_provider_used_when_primary_exhausted():
    s = Settings(llm_max_retries=1, llm_backoff_seconds=0, openai_api_key="k", llm_provider="ollama", llm_fallback_provider="openai")

    def handler(req):
        if "11434" in str(req.url):
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"from": "fallback"}'}}]})

    llm = build_llm(s, transport=httpx.MockTransport(handler))
    assert isinstance(llm, FallbackLLM)
    with request_stats() as stats:
        assert await llm.generate_json(system="s", user="u") == {"from": "fallback"}
    assert stats.fallbacks == 1 and llm.describe()["fallback"]["provider"] == "openai-compatible"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        build_llm(Settings(llm_provider="skynet"))

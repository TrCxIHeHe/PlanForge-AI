import pytest

from app.config import Settings
from scripts import check_gemini
from scripts.check_gemini import available_text_models, list_models, save_model_to_env, select_text_model


def test_settings_load_gemini_api_key_without_openai_key_alias(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("GEMINI_API_KEY", "local-only-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env()

    assert settings.gemini_api_key == "local-only-key"
    assert settings.openai_api_key == ""


def test_select_text_model_uses_preferred_when_account_supports_generate_content():
    models = [
        {"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
    ]

    assert select_text_model(models, "gemini-3.8-flash") == "gemini-3.8-flash"


def test_select_text_model_ignores_models_without_text_generation():
    models = [
        {"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateSpeech"]},
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    ]

    assert select_text_model(models, "gemini-3.8-flash") == "gemini-2.5-flash"
    assert available_text_models(models) == ["gemini-2.5-flash"]


def test_select_text_model_reports_no_compatible_model():
    with pytest.raises(RuntimeError, match="no model supporting generateContent"):
        select_text_model([{"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateSpeech"]}], "gemini-3.8-flash")


def test_save_model_to_env_only_changes_model_setting(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=local-secret\nOPENAI_MODEL=old-model\n", encoding="utf-8")

    save_model_to_env(env_path, "gemini-2.5-flash")

    assert env_path.read_text(encoding="utf-8") == "GEMINI_API_KEY=local-secret\nOPENAI_MODEL=gemini-2.5-flash\n"


def test_list_models_sends_gemini_api_key_not_openai_key(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": []}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr("scripts.check_gemini.httpx.get", fake_get)
    settings = Settings(
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openai_api_key="legacy-key",
        gemini_api_key="gemini-local-key",
    )

    assert list_models(settings) == []
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models"
    assert captured["headers"] == {"x-goog-api-key": "gemini-local-key"}


async def test_gemini_check_does_not_require_legacy_openai_key(monkeypatch):
    settings = Settings(
        llm_provider="openai",
        llm_fallback_provider="ollama",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openai_model="gemini-3.8-flash",
        gemini_api_key="gemini-local-key",
    )
    monkeypatch.setattr(check_gemini.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        check_gemini,
        "list_models",
        lambda _settings: [{
            "name": "models/gemini-3.8-flash",
            "supportedGenerationMethods": ["generateContent"],
        }],
    )

    async def probe(_settings):
        return None

    monkeypatch.setattr(check_gemini, "_probe_gemini", probe)

    assert await check_gemini.main() == 0

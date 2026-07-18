"""Unit tests for the provider abstraction layer.

All external SDK calls (openai, anthropic, google-generativeai, ollama) are
mocked; no real network calls are made. Each provider module optionally
imports its vendor SDK (falling back to ``None`` when not installed), so
tests monkeypatch the module-level SDK references directly rather than
relying on the real packages being present.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from providers.base import (
    LLMRequest,
    LLMResponse,
    ModelNotFoundError,
    ProviderError,
    RateLimitedError,
    TransientProviderError,
    retry_with_backoff,
)
from providers.factory import ProviderFactory
from providers.registry import ProviderRegistry

import providers.anthropic_provider as anthropic_provider_mod
import providers.gemini_provider as gemini_provider_mod
import providers.ollama_provider as ollama_provider_mod
import providers.openai_provider as openai_provider_mod


# --------------------------------------------------------------------------- #
# Shared fixtures / fakes
# --------------------------------------------------------------------------- #


class _FakeAPIError(Exception):
    """Generic stand-in base for vendor SDK exceptions."""

    def __init__(self, message: str = "error", status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


def _make_fake_exception_namespace() -> MagicMock:
    """Build a fake SDK module namespace exposing real Exception subclasses.

    The provider modules reference exception classes as attributes on the
    vendor module (e.g. ``openai.RateLimitError``). For ``except`` clauses to
    work at runtime, these must be real exception *classes*, so we can't use
    plain MagicMock instances here.
    """
    ns = MagicMock()
    ns.NotFoundError = type("NotFoundError", (_FakeAPIError,), {})
    ns.RateLimitError = type("RateLimitError", (_FakeAPIError,), {})
    ns.APIConnectionError = type("APIConnectionError", (_FakeAPIError,), {})
    ns.APITimeoutError = type("APITimeoutError", (_FakeAPIError,), {})
    ns.InternalServerError = type("InternalServerError", (_FakeAPIError,), {})
    ns.APIStatusError = type("APIStatusError", (_FakeAPIError,), {})
    return ns


@pytest.fixture
def request_obj() -> LLMRequest:
    return LLMRequest(system_prompt="You are helpful.", user_prompt="Say hi.", temperature=0.5, max_tokens=64)


# --------------------------------------------------------------------------- #
# LLMRequest / LLMResponse model tests
# --------------------------------------------------------------------------- #


class TestModels:
    def test_llm_request_defaults(self) -> None:
        req = LLMRequest(user_prompt="hello")
        assert req.system_prompt is None
        assert req.temperature == 0.3
        assert req.max_tokens == 1024

    def test_llm_request_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValueError):
            LLMRequest(user_prompt="   ")

    def test_llm_request_rejects_out_of_range_temperature(self) -> None:
        with pytest.raises(ValueError):
            LLMRequest(user_prompt="hi", temperature=5.0)

    def test_llm_response_defaults(self) -> None:
        resp = LLMResponse(text="hi", latency_ms=12.3)
        assert resp.token_usage == {}
        assert resp.finish_reason == "unknown"
        assert resp.raw_metadata == {}


# --------------------------------------------------------------------------- #
# retry_with_backoff tests
# --------------------------------------------------------------------------- #


class TestRetryWithBackoff:
    def test_succeeds_without_retry(self) -> None:
        calls = {"n": 0}

        @retry_with_backoff((TransientProviderError,), max_attempts=3, base_delay=0.001)
        def fn() -> str:
            calls["n"] += 1
            return "ok"

        assert fn() == "ok"
        assert calls["n"] == 1

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        @retry_with_backoff((TransientProviderError,), max_attempts=3, base_delay=0.001)
        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientProviderError("temporary")
            return "ok"

        assert fn() == "ok"
        assert calls["n"] == 3

    def test_exhausts_attempts_and_raises(self) -> None:
        calls = {"n": 0}

        @retry_with_backoff((TransientProviderError,), max_attempts=2, base_delay=0.001)
        def fn() -> str:
            calls["n"] += 1
            raise TransientProviderError("always fails")

        with pytest.raises(TransientProviderError):
            fn()
        assert calls["n"] == 2

    def test_non_retryable_exception_propagates_immediately(self) -> None:
        calls = {"n": 0}

        @retry_with_backoff((TransientProviderError,), max_attempts=3, base_delay=0.001)
        def fn() -> str:
            calls["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            fn()
        assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# OpenAIProvider tests
# --------------------------------------------------------------------------- #


class TestOpenAIProvider:
    def _build_provider(self, monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> Any:
        fake_openai = _make_fake_exception_namespace()
        monkeypatch.setattr(openai_provider_mod, "openai", fake_openai)
        fake_openai_cls = MagicMock(return_value=client)
        monkeypatch.setattr(openai_provider_mod, "OpenAI", fake_openai_cls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return openai_provider_mod.OpenAIProvider(model="gpt-4o")

    def _fake_completion(self, text: str = "Hello!", finish_reason: str = "stop") -> MagicMock:
        completion = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        choice.finish_reason = finish_reason
        completion.choices = [choice]
        completion.usage.prompt_tokens = 10
        completion.usage.completion_tokens = 5
        completion.usage.total_tokens = 15
        completion.model_dump.return_value = {"id": "chatcmpl-123"}
        return completion

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_openai = _make_fake_exception_namespace()
        monkeypatch.setattr(openai_provider_mod, "openai", fake_openai)
        monkeypatch.setattr(openai_provider_mod, "OpenAI", MagicMock())
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderError):
            openai_provider_mod.OpenAIProvider(model="gpt-4o")

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = self._fake_completion()
        provider = self._build_provider(monkeypatch, client)

        response = provider.generate(request_obj)

        assert isinstance(response, LLMResponse)
        assert response.text == "Hello!"
        assert response.finish_reason == "stop"
        assert response.token_usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        client.chat.completions.create.assert_called_once()
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert kwargs["messages"][1] == {"role": "user", "content": "Say hi."}

    def test_model_not_found_raises(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        provider = self._build_provider(monkeypatch, client)
        client.chat.completions.create.side_effect = openai_provider_mod.openai.NotFoundError("no model")

        with pytest.raises(ModelNotFoundError):
            provider.generate(request_obj)

    def test_rate_limited_retries_then_raises(
        self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest
    ) -> None:
        client = MagicMock()
        provider = self._build_provider(monkeypatch, client)
        client.chat.completions.create.side_effect = openai_provider_mod.openai.RateLimitError("slow down")

        with pytest.raises(RateLimitedError):
            provider.generate(request_obj)
        assert client.chat.completions.create.call_count == 3  # default max_attempts

    def test_transient_error_recovers_on_retry(
        self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest
    ) -> None:
        client = MagicMock()
        provider = self._build_provider(monkeypatch, client)
        client.chat.completions.create.side_effect = [
            openai_provider_mod.openai.APIConnectionError("net blip"),
            self._fake_completion(text="Recovered"),
        ]

        response = provider.generate(request_obj)
        assert response.text == "Recovered"
        assert client.chat.completions.create.call_count == 2


# --------------------------------------------------------------------------- #
# AnthropicProvider tests
# --------------------------------------------------------------------------- #


class TestAnthropicProvider:
    def _build_provider(self, monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> Any:
        fake_anthropic = _make_fake_exception_namespace()
        fake_anthropic.Anthropic = MagicMock(return_value=client)
        monkeypatch.setattr(anthropic_provider_mod, "anthropic", fake_anthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        return anthropic_provider_mod.AnthropicProvider(model="claude-sonnet-4-6")

    def _fake_message(self, text: str = "Hello there!", stop_reason: str = "end_turn") -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        message = MagicMock()
        message.content = [block]
        message.usage.input_tokens = 8
        message.usage.output_tokens = 4
        message.stop_reason = stop_reason
        message.model_dump.return_value = {"id": "msg_123"}
        return message

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_anthropic = _make_fake_exception_namespace()
        fake_anthropic.Anthropic = MagicMock()
        monkeypatch.setattr(anthropic_provider_mod, "anthropic", fake_anthropic)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderError):
            anthropic_provider_mod.AnthropicProvider(model="claude-sonnet-4-6")

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        client.messages.create.return_value = self._fake_message()
        provider = self._build_provider(monkeypatch, client)

        response = provider.generate(request_obj)

        assert response.text == "Hello there!"
        assert response.finish_reason == "end_turn"
        assert response.token_usage == {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
        }
        _, kwargs = client.messages.create.call_args
        assert kwargs["system"] == "You are helpful."
        assert kwargs["messages"] == [{"role": "user", "content": "Say hi."}]

    def test_model_not_found_raises(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        provider = self._build_provider(monkeypatch, client)
        client.messages.create.side_effect = anthropic_provider_mod.anthropic.NotFoundError("no model")

        with pytest.raises(ModelNotFoundError):
            provider.generate(request_obj)

    def test_transient_error_recovers_on_retry(
        self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest
    ) -> None:
        client = MagicMock()
        provider = self._build_provider(monkeypatch, client)
        client.messages.create.side_effect = [
            anthropic_provider_mod.anthropic.InternalServerError("5xx"),
            self._fake_message(text="Recovered"),
        ]

        response = provider.generate(request_obj)
        assert response.text == "Recovered"
        assert client.messages.create.call_count == 2


# --------------------------------------------------------------------------- #
# GeminiProvider tests
# --------------------------------------------------------------------------- #


class TestGeminiProvider:
    def _build_provider(self, monkeypatch: pytest.MonkeyPatch, model_instance: MagicMock) -> Any:
        fake_google_exceptions = _make_fake_exception_namespace()
        fake_google_exceptions.NotFound = fake_google_exceptions.NotFoundError
        fake_google_exceptions.ResourceExhausted = fake_google_exceptions.RateLimitError
        fake_google_exceptions.DeadlineExceeded = fake_google_exceptions.APITimeoutError
        fake_google_exceptions.ServiceUnavailable = fake_google_exceptions.APIConnectionError
        fake_google_exceptions.GoogleAPIError = type("GoogleAPIError", (_FakeAPIError,), {})
        monkeypatch.setattr(gemini_provider_mod, "google_exceptions", fake_google_exceptions)

        fake_genai = MagicMock()
        fake_genai.GenerativeModel = MagicMock(return_value=model_instance)
        fake_genai.types.GenerationConfig = MagicMock(side_effect=lambda **kw: kw)
        monkeypatch.setattr(gemini_provider_mod, "genai", fake_genai)

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        provider = gemini_provider_mod.GeminiProvider(model="gemini-1.5-pro")
        return provider, fake_genai

    def _fake_result(self, text: str = "Hi from Gemini") -> MagicMock:
        result = MagicMock()
        result.text = text
        result.usage_metadata.prompt_token_count = 6
        result.usage_metadata.candidates_token_count = 3
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        result.candidates = [candidate]
        result.to_dict.return_value = {"raw": "gemini-response"}
        return result

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gemini_provider_mod, "genai", MagicMock())
        monkeypatch.setattr(gemini_provider_mod, "google_exceptions", MagicMock())
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ProviderError):
            gemini_provider_mod.GeminiProvider(model="gemini-1.5-pro")

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        model_instance = MagicMock()
        model_instance.generate_content.return_value = self._fake_result()
        provider, fake_genai = self._build_provider(monkeypatch, model_instance)

        response = provider.generate(request_obj)

        assert response.text == "Hi from Gemini"
        assert response.token_usage == {
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "total_tokens": 9,
        }
        assert response.finish_reason == "STOP"
        # A system prompt was provided, so a fresh model must be built with it.
        fake_genai.GenerativeModel.assert_any_call(
            model_name="gemini-1.5-pro", system_instruction="You are helpful."
        )

    def test_model_not_found_raises(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        model_instance = MagicMock()
        provider, _ = self._build_provider(monkeypatch, model_instance)
        model_instance.generate_content.side_effect = gemini_provider_mod.google_exceptions.NotFound("nope")

        with pytest.raises(ModelNotFoundError):
            provider.generate(request_obj)

    def test_rate_limited_raises(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        model_instance = MagicMock()
        provider, _ = self._build_provider(monkeypatch, model_instance)
        model_instance.generate_content.side_effect = gemini_provider_mod.google_exceptions.ResourceExhausted(
            "quota"
        )

        with pytest.raises(RateLimitedError):
            provider.generate(request_obj)


# --------------------------------------------------------------------------- #
# OllamaProvider tests
# --------------------------------------------------------------------------- #


class TestOllamaProvider:
    def _build_provider(self, monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> Any:
        fake_ollama = MagicMock()
        monkeypatch.setattr(ollama_provider_mod, "ollama", fake_ollama)
        # No host supplied -> provider uses the top-level `ollama` module as client.
        fake_ollama.chat = client.chat
        return ollama_provider_mod.OllamaProvider(model="llama3.1")

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        client.chat.return_value = {
            "message": {"content": "Hi from Ollama"},
            "prompt_eval_count": 20,
            "eval_count": 10,
            "done": True,
        }
        provider = self._build_provider(monkeypatch, client)

        response = provider.generate(request_obj)

        assert response.text == "Hi from Ollama"
        assert response.finish_reason == "stop"
        assert response.token_usage == {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }

    def test_model_not_found_raises(self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest) -> None:
        client = MagicMock()
        client.chat.side_effect = Exception("model 'llama3.1' not found, try pulling it")
        provider = self._build_provider(monkeypatch, client)

        with pytest.raises(ModelNotFoundError):
            provider.generate(request_obj)

    def test_generic_error_wrapped_as_transient(
        self, monkeypatch: pytest.MonkeyPatch, request_obj: LLMRequest
    ) -> None:
        client = MagicMock()
        client.chat.side_effect = ConnectionError("daemon unreachable")
        provider = self._build_provider(monkeypatch, client)

        with pytest.raises(TransientProviderError):
            provider.generate(request_obj)

    def test_missing_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ollama_provider_mod, "ollama", None)
        with pytest.raises(ImportError):
            ollama_provider_mod.OllamaProvider(model="llama3.1")

    def test_explicit_host_uses_client_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_ollama = MagicMock()
        fake_client_instance = MagicMock()
        fake_client_instance.chat.return_value = {
            "message": {"content": "remote hi"},
            "prompt_eval_count": 1,
            "eval_count": 1,
            "done": True,
        }
        fake_ollama.Client = MagicMock(return_value=fake_client_instance)
        monkeypatch.setattr(ollama_provider_mod, "ollama", fake_ollama)

        provider = ollama_provider_mod.OllamaProvider(model="llama3.1", host="http://remote:11434")
        response = provider.generate(LLMRequest(user_prompt="hi"))

        fake_ollama.Client.assert_called_once_with(host="http://remote:11434")
        assert response.text == "remote hi"


# --------------------------------------------------------------------------- #
# ProviderFactory tests
# --------------------------------------------------------------------------- #


class TestProviderFactory:
    def test_supported_providers(self) -> None:
        assert ProviderFactory.supported_providers() == ["anthropic", "gemini", "ollama", "openai"]

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ProviderError):
            ProviderFactory.create("does-not-exist", model="foo")

    def test_case_insensitive_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_ollama = MagicMock()
        monkeypatch.setattr(ollama_provider_mod, "ollama", fake_ollama)
        provider = ProviderFactory.create(" OLLAMA ", model="llama3.1")
        assert isinstance(provider, ollama_provider_mod.OllamaProvider)
        assert provider.model == "llama3.1"

    def test_create_openai_via_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_openai = _make_fake_exception_namespace()
        monkeypatch.setattr(openai_provider_mod, "openai", fake_openai)
        monkeypatch.setattr(openai_provider_mod, "OpenAI", MagicMock(return_value=MagicMock()))
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        provider = ProviderFactory.create("openai", model="gpt-4o")
        assert isinstance(provider, openai_provider_mod.OpenAIProvider)


# --------------------------------------------------------------------------- #
# ProviderRegistry tests
# --------------------------------------------------------------------------- #


class TestProviderRegistry:
    def test_list_providers_reports_configured_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "set")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        providers = {p["name"]: p for p in ProviderRegistry.list_providers()}

        assert providers["ollama"]["configured"] is True
        assert providers["ollama"]["requires_api_key"] is False
        assert providers["openai"]["configured"] is True
        assert providers["anthropic"]["configured"] is False
        assert providers["gemini"]["configured"] is False

    def test_detect_ollama_models_returns_empty_when_package_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import providers.registry as registry_mod

        monkeypatch.setattr(registry_mod, "ollama", None)
        assert ProviderRegistry.detect_ollama_models() == []

    def test_detect_ollama_models_parses_dict_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import providers.registry as registry_mod

        fake_ollama = MagicMock()
        fake_ollama.list.return_value = {"models": [{"name": "llama3.1"}, {"model": "mistral:latest"}]}
        monkeypatch.setattr(registry_mod, "ollama", fake_ollama)

        models = ProviderRegistry.detect_ollama_models()
        assert models == ["llama3.1", "mistral:latest"]

    def test_detect_ollama_models_handles_daemon_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import providers.registry as registry_mod

        fake_ollama = MagicMock()
        fake_ollama.list.side_effect = ConnectionError("daemon not running")
        monkeypatch.setattr(registry_mod, "ollama", fake_ollama)

        assert ProviderRegistry.detect_ollama_models() == []
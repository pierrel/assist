"""Unit tests for the auto-discovery / cache / cache-buster machinery.

No LLM, no network.  All HTTP calls are intercepted via monkeypatch on
``httpx.get``; the conftest autouse fixture stubs ``_probe_endpoint`` for
tests that don't override it explicitly.
"""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

import openai

from assist import model_manager
from assist.model_manager import (
    OpenAIConfig,
    _ModelNotFoundCacheBuster,
    _resolve_api_key,
    invalidate_config_cache,
)


# The conftest autouse fixture monkeypatches ``_probe_endpoint`` on the
# module.  Capture the real function here, before the fixture runs, so
# ``TestProbeEndpoint`` can exercise it against a faked ``httpx.get``.
_real_probe_endpoint = model_manager._probe_endpoint


# Canonical vLLM /v1/models payload shape.
_VLLM_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 53616,
        }
    ],
}

# Canonical llama.cpp /v1/models payload — note the absence of
# ``max_model_len`` and the presence of ``meta.n_ctx_train`` (the
# trained length, which we deliberately ignore — the runtime context
# can be smaller than the trained one).
_LLAMACPP_MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "Qwen_Qwen3.6-27B-Q4_K_M.gguf",
            "object": "model",
            "owned_by": "llamacpp",
            "meta": {
                "n_ctx_train": 262144,
                "n_vocab": 248320,
            },
        }
    ],
}

# Canonical llama.cpp /props payload — the runtime context length lives
# at ``default_generation_settings.n_ctx``.  Per-slot value, which is
# what a single conversation can use (llama.cpp divides ``-c`` across
# ``--parallel`` slots).
_LLAMACPP_PROPS_PAYLOAD = {
    "default_generation_settings": {
        "params": {"temperature": 1.0},
        "n_ctx": 200192,
    },
    "total_slots": 1,
    "model_alias": "Qwen_Qwen3.6-27B-Q4_K_M.gguf",
    "model_path": "/path/to/model.gguf",
}


def _make_response(payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    return response


class TestProbeEndpoint(TestCase):
    """Behavior of the live probe — uses ``_real_probe_endpoint`` to
    sidestep the conftest stub, with a faked ``httpx.get`` underneath.
    """

    def test_extracts_model_and_context_len(self):
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response(_VLLM_PAYLOAD)
            config = _real_probe_endpoint("http://test:8000/v1", "EMPTY")
        self.assertEqual(
            config.model, "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
        )
        self.assertEqual(config.context_len, 53616)
        self.assertEqual(config.url, "http://test:8000/v1")

    def test_picks_first_model_when_multiple(self):
        payload = {
            "data": [
                {"id": "model-a", "max_model_len": 1000},
                {"id": "model-b", "max_model_len": 2000},
            ]
        }
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response(payload)
            config = _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertEqual(config.model, "model-a")

    def test_handles_missing_max_model_len(self):
        payload = {"data": [{"id": "real-openai-model"}]}
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response(payload)
            with self.assertLogs("assist.model_manager", level="WARNING"):
                config = _real_probe_endpoint("http://x/v1", "k")
        self.assertEqual(config.context_len, 32768)

    def test_raises_on_unreachable_endpoint(self):
        import httpx as _httpx

        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.side_effect = _httpx.ConnectError("connection refused")
            with self.assertRaises(RuntimeError) as ctx:
                _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertIn("Could not reach", str(ctx.exception))

    def test_raises_on_empty_data(self):
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response({"data": []})
            with self.assertRaises(RuntimeError) as ctx:
                _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertIn("returned no models", str(ctx.exception))

    def test_strips_trailing_slash_on_url(self):
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response(_VLLM_PAYLOAD)
            _real_probe_endpoint("http://x/v1/", "EMPTY")
            (called_url,), _ = fake_get.call_args
        self.assertEqual(called_url, "http://x/v1/models")

    def test_falls_back_to_props_when_max_model_len_missing(self):
        """llama.cpp omits ``max_model_len`` from /v1/models. The runtime
        context lives at ``default_generation_settings.n_ctx`` on /props
        (server root, not under /v1).  We probe both and use the /props
        value rather than the trained-length fallback.
        """
        responses = [
            _make_response(_LLAMACPP_MODELS_PAYLOAD),
            _make_response(_LLAMACPP_PROPS_PAYLOAD),
        ]
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.side_effect = responses
            config = _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertEqual(config.model, "Qwen_Qwen3.6-27B-Q4_K_M.gguf")
        self.assertEqual(config.context_len, 200192)
        # Two HTTP calls: /v1/models then /props (root-level).
        called = [c.args[0] for c in fake_get.call_args_list]
        self.assertEqual(called, ["http://x/v1/models", "http://x/props"])

    def test_props_url_strips_v1_suffix(self):
        """Variants of ASSIST_MODEL_URL: with/without trailing slash, with
        or without /v1.  /props always lives at the server root.
        """
        responses_template = [
            _make_response(_LLAMACPP_MODELS_PAYLOAD),
            _make_response(_LLAMACPP_PROPS_PAYLOAD),
        ]
        for input_url in ("http://x/v1", "http://x/v1/", "http://x"):
            with patch("assist.model_manager.httpx.get") as fake_get:
                fake_get.side_effect = list(responses_template)
                _real_probe_endpoint(input_url, "EMPTY")
                called = [c.args[0] for c in fake_get.call_args_list]
            self.assertEqual(
                called[1], "http://x/props",
                f"For ASSIST_MODEL_URL={input_url!r} the /props probe "
                f"must hit the server root, got {called[1]!r}",
            )

    def test_falls_back_to_default_when_both_probes_fail(self):
        """If /v1/models has no max_model_len AND /props is unreachable
        or returns junk, fall back to the 32768 default and warn.  This
        is the "neither vLLM nor llama.cpp surfaced anything" path.
        """
        import httpx as _httpx

        responses = [
            _make_response(_LLAMACPP_MODELS_PAYLOAD),
        ]
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.side_effect = (
                list(responses)
                + [_httpx.ConnectError("connection refused")]
            )
            with self.assertLogs("assist.model_manager", level="WARNING"):
                config = _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertEqual(config.context_len, 32768)

    def test_props_payload_missing_nested_key_falls_back(self):
        """If /props returns 200 but ``default_generation_settings.n_ctx``
        is missing (future schema change, partial response), fall back to
        the default.  Don't propagate the ambiguous "trained ctx" from
        the llama.cpp /v1/models entry.
        """
        responses = [
            _make_response(_LLAMACPP_MODELS_PAYLOAD),
            _make_response({"default_generation_settings": {}}),
        ]
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.side_effect = responses
            with self.assertLogs("assist.model_manager", level="WARNING"):
                config = _real_probe_endpoint("http://x/v1", "EMPTY")
        self.assertEqual(config.context_len, 32768)

    def test_does_not_probe_props_when_max_model_len_present(self):
        """vLLM's happy path: ``max_model_len`` already in /v1/models.
        Don't bother calling /props — saves an HTTP round-trip and means
        non-llama.cpp servers without /props support don't see noise.
        """
        with patch("assist.model_manager.httpx.get") as fake_get:
            fake_get.return_value = _make_response(_VLLM_PAYLOAD)
            _real_probe_endpoint("http://x/v1", "EMPTY")
            self.assertEqual(fake_get.call_count, 1)


class TestGetConfigCache(TestCase):
    """Caching + invalidation behavior of ``_get_config``."""

    def test_returns_none_when_url_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(model_manager._get_config())

    def test_invalidate_forces_reprobe(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            first = model_manager._get_config()
            invalidate_config_cache()
            second = model_manager._get_config()
        self.assertIsNot(first, second)
        self.assertEqual(first, second)

    def test_url_change_busts_cache(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://a/v1"}):
            first = model_manager._get_config()
            cached = model_manager._get_config()
            self.assertIs(first, cached)
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://b/v1"}):
            second = model_manager._get_config()
        self.assertEqual(first.url, "http://a/v1")
        self.assertEqual(second.url, "http://b/v1")


class TestApiKeyFallback(TestCase):
    """ASSIST_API_KEY → OPENAI_API_KEY → ``"EMPTY"``."""

    def test_uses_assist_key_when_set(self):
        with patch.dict(
            "os.environ",
            {"ASSIST_API_KEY": "assist-key", "OPENAI_API_KEY": "openai-key"},
            clear=True,
        ):
            self.assertEqual(_resolve_api_key(), "assist-key")

    def test_falls_back_to_openai_key(self):
        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "openai-key"}, clear=True
        ):
            self.assertEqual(_resolve_api_key(), "openai-key")

    def test_falls_back_to_empty_string(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_api_key(), "EMPTY")


def _make_not_found(code: str | None) -> openai.NotFoundError:
    """Build a real ``openai.NotFoundError`` with the given ``code``.

    The SDK extracts ``code`` from the response body during
    ``APIError.__init__``, so we have to construct it through the SDK
    constructor rather than monkey-patching attributes.
    """
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    response.request = MagicMock()
    body = {"code": code} if code is not None else None
    return openai.NotFoundError(
        message="model not found",
        response=response,
        body=body,
    )


class TestCacheBusterCallback(TestCase):
    """``_ModelNotFoundCacheBuster.on_llm_error`` behavior."""

    def test_invalidates_on_model_not_found(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            model_manager._get_config()
            self.assertIsNotNone(model_manager._cached_config)

            buster = _ModelNotFoundCacheBuster()
            with self.assertLogs("assist.model_manager", level="WARNING"):
                buster.on_llm_error(_make_not_found("model_not_found"))
            self.assertIsNone(model_manager._cached_config)

    def test_passes_through_other_not_found_codes(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            model_manager._get_config()
            buster = _ModelNotFoundCacheBuster()
            buster.on_llm_error(_make_not_found("some_other_code"))
            self.assertIsNotNone(model_manager._cached_config)

    def test_passes_through_unrelated_exceptions(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            model_manager._get_config()
            buster = _ModelNotFoundCacheBuster()
            buster.on_llm_error(ValueError("unrelated"))
            self.assertIsNotNone(model_manager._cached_config)


class TestSelectChatModel(TestCase):
    """End-to-end behavior of ``select_chat_model``."""

    def test_raises_when_url_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                model_manager.select_chat_model(0.1)
        self.assertIn("ASSIST_MODEL_URL", str(ctx.exception))

    def test_attaches_max_input_tokens_to_profile(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            llm = model_manager.select_chat_model(0.1)
        self.assertEqual(llm.profile["max_input_tokens"], 32768)

    def test_attaches_cache_buster_callback(self):
        with patch.dict("os.environ", {"ASSIST_MODEL_URL": "http://x/v1"}):
            llm = model_manager.select_chat_model(0.1)
        callbacks = llm.callbacks or []
        self.assertTrue(
            any(isinstance(cb, _ModelNotFoundCacheBuster) for cb in callbacks),
            f"Expected _ModelNotFoundCacheBuster in callbacks; got {callbacks!r}",
        )

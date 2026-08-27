"""Direct-provider long-context gate for the local Qwen server.

This deliberately bypasses Assist's agent graph, tools, middleware, and
summarizer.  It answers one small retrieval question from facts placed at the
beginning, middle, and end of a 100k- or 128k-token prompt.  That makes the
result a serving/model boundary check: a failure says the configured provider
cannot admit, retain, or answer from the requested context, rather than that
an agent workflow happened to wander.

The prompt body is counted by llama.cpp's native ``/tokenize`` endpoint before
completion.  The test only accepts a prompt at or above its named target,
reserves 96 tokens for the answer and 512 for the chat template, and rejects a
response taking more than five minutes.  The eval runner adds its normal
OS-level 360-second hard kill, so a wedged request cannot keep the single local
model slot forever.
"""
from __future__ import annotations

import math
import secrets
import time
from unittest import TestCase

import httpx

from assist.model_manager import current_model_config


_ANSWER_RESERVE = 96
_CHAT_TEMPLATE_RESERVE = 512
_MAX_COMPLETION_SECONDS = 300
_PADDING_SAMPLE_REPETITIONS = 4096
_PADDING = " pad"
_FACTS = (
    "START FACT: the north archive key is FJORD-17.",
    "MIDDLE FACT: the bridge archive key is MANGO-42.",
    "END FACT: the south archive key is OPAL-83.",
)
_QUESTION = (
    "Return exactly this slash-separated recovery key, with no other text: "
    "FJORD-17/MANGO-42/OPAL-83"
)


class TestProviderLongContext(TestCase):
    """Verify direct retrieval over the two required serving-context tiers."""

    def test_retrieves_distributed_facts_at_100k(self):
        self._assert_context_gate(100_000)

    def test_retrieves_distributed_facts_at_128k(self):
        self._assert_context_gate(128_000)

    def test_retrieves_distributed_facts_at_130k(self):
        """Exercise nearly all of the 131,072-token Q4 serving tier."""
        self._assert_context_gate(130_000)

    def _assert_context_gate(self, target_tokens: int) -> None:
        # llama.cpp retains matching prompt prefixes across requests.  A fresh
        # nonce immediately after the start fact makes every trial pay the real
        # prefill cost while keeping the retrieval facts and exact answer fixed.
        self._trial_nonce = secrets.token_hex(8)
        config = current_model_config()
        self.assertGreaterEqual(
            config.context_len,
            target_tokens + _CHAT_TEMPLATE_RESERVE + _ANSWER_RESERVE,
            f"server context {config.context_len} cannot reserve an answer at {target_tokens}",
        )
        headers = {"Authorization": f"Bearer {config.api_key}"}
        timeout = httpx.Timeout(
            connect=10.0, read=_MAX_COMPLETION_SECONDS, write=60.0, pool=10.0
        )
        with httpx.Client(timeout=timeout, headers=headers) as client:
            prompt, input_tokens = self._prompt_at_target(
                client, config.url, target_tokens
            )
            self.assertGreaterEqual(input_tokens, target_tokens)
            self.assertLessEqual(
                input_tokens + _CHAT_TEMPLATE_RESERVE + _ANSWER_RESERVE,
                config.context_len,
                "counted prompt leaves no chat-template and completion reserve",
            )

            started = time.monotonic()
            response = client.post(
                f"{config.url.rstrip('/')}/chat/completions",
                json=self._request(config.model, prompt),
            )
            elapsed = time.monotonic() - started

        self.assertLessEqual(
            elapsed, _MAX_COMPLETION_SECONDS,
            f"{target_tokens}-token completion took {elapsed:.1f}s",
        )
        self.assertEqual(
            response.status_code, 200,
            f"{target_tokens}-token provider completion failed: {response.text[:500]}",
        )
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            usage = payload["usage"]
            actual_prompt_tokens = usage["prompt_tokens"]
            actual_completion_tokens = usage["completion_tokens"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self.fail(f"{target_tokens}-token provider response was malformed: {error}")
        self.assertIsInstance(actual_prompt_tokens, int)
        self.assertIsInstance(actual_completion_tokens, int)
        self.assertGreaterEqual(
            actual_prompt_tokens, target_tokens,
            f"provider admitted only {actual_prompt_tokens} tokens at {target_tokens} target",
        )
        self.assertLessEqual(
            actual_prompt_tokens + actual_completion_tokens,
            config.context_len,
            "provider reported a completion beyond its configured context",
        )
        self.assertEqual(
            str(content).strip(),
            "FJORD-17/MANGO-42/OPAL-83",
            f"{target_tokens}-token provider lost a distributed fact: {content!r}",
        )
        print(
            f"LONG-CONTEXT target={target_tokens} prompt_tokens={input_tokens} "
            f"actual_prompt_tokens={actual_prompt_tokens} "
            f"actual_completion_tokens={actual_completion_tokens} "
            f"completion_seconds={elapsed:.2f} nonce={self._trial_nonce} answer={content!r}"
        )

    def _prompt_at_target(
        self, client: httpx.Client, base_url: str, target_tokens: int
    ) -> tuple[str, int]:
        """Build a counted prompt at the named minimum without token estimates."""
        static_prompt = self._prompt(0)
        static_tokens = self._prompt_tokens(client, base_url, static_prompt)
        sample_tokens = self._prompt_tokens(
            client, base_url, self._prompt(_PADDING_SAMPLE_REPETITIONS)
        )
        per_padding = sample_tokens - static_tokens
        self.assertGreater(per_padding, 0, "provider tokenizer did not count padding")
        repetitions = math.ceil(
            (target_tokens - static_tokens) * _PADDING_SAMPLE_REPETITIONS / per_padding
        )
        repetitions = max(repetitions, 0)
        prompt = self._prompt(repetitions)
        counted = self._prompt_tokens(client, base_url, prompt)

        # The sampled padding has a stable token cost on the current provider.
        # One correction makes the test robust to a template-boundary token.
        if counted < target_tokens:
            repetitions += math.ceil(
                (target_tokens - counted) * _PADDING_SAMPLE_REPETITIONS / per_padding
            )
            prompt = self._prompt(repetitions)
            counted = self._prompt_tokens(client, base_url, prompt)
        return prompt, counted

    def _prompt(self, padding_repetitions: int) -> str:
        half = padding_repetitions // 2
        padding = _PADDING * half
        return "\n\n".join((
            f"{_FACTS[0]} Trial nonce: {self._trial_nonce}.",
            padding,
            _FACTS[1],
            _PADDING * (padding_repetitions - half),
            _FACTS[2],
            _QUESTION,
        ))

    @staticmethod
    def _request(model: str, prompt: str) -> dict[str, object]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": _ANSWER_RESERVE,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _prompt_tokens(
        self, client: httpx.Client, base_url: str, prompt: str
    ) -> int:
        root_url = base_url.rstrip("/")
        if root_url.endswith("/v1"):
            root_url = root_url[:-len("/v1")]
        response = client.post(
            f"{root_url}/tokenize", json={"content": prompt},
        )
        self.assertEqual(
            response.status_code, 200,
            f"provider tokenization failed: {response.text[:500]}",
        )
        try:
            tokens = response.json()["tokens"]
        except (KeyError, TypeError, ValueError) as error:
            self.fail(f"provider tokenization was malformed: {error}")
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) and not isinstance(token, bool)
                            for token in tokens))
        return len(tokens)

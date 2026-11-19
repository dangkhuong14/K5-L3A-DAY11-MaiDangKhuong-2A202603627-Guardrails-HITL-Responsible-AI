"""Claude routing tests; no real keys or network calls."""
import asyncio
import os
import unittest
from unittest.mock import patch

from agents.red_advance_nemo import build_rails_dict
from core import config
from core.openai_runtime import create_openai_pair


class AnthropicProviderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "RED_TEAM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "fake-anthropic",
            "OPENAI_API_KEY": "fake-openai",
            "BLUE_TEAM_PROVIDER": "liquid",
            "LIQUID_API_KEY": "fake-liquid",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_provider_alias_and_model_override(self):
        os.environ["RED_TEAM_PROVIDER"] = "claude"
        os.environ["ANTHROPIC_MODEL"] = "test-claude-model"
        self.assertEqual(config.get_red_provider(), "anthropic")
        self.assertTrue(config.red_uses_openai_sdk())
        self.assertFalse(config.red_uses_gemini())
        self.assertEqual(config.get_red_model_default(), "test-claude-model")
        self.assertEqual(config.get_red_model_advance(), "test-claude-model")

    def test_runner_and_nemo_use_same_endpoint_model_key_and_workspace(self):
        os.environ.update(ANTHROPIC_BASE_URL="https://anthropic.example/v1/",
                          ANTHROPIC_WORKSPACE_ID="workspace-test")
        agent, runner = create_openai_pair(name="red", instruction="Banking", app_name="test")
        entry = build_rails_dict()["models"][0]
        self.assertEqual(agent.provider, "anthropic")
        self.assertEqual(runner.model, "claude-sonnet-4-6")
        self.assertEqual(runner.model, entry["model"])
        self.assertEqual(entry["engine"], "openai")
        self.assertEqual(entry["api_key_env_var"], "ANTHROPIC_API_KEY")
        self.assertEqual(runner.client_kwargs, {"api_key": "fake-anthropic", **entry["parameters"]})
        self.assertEqual(entry["parameters"]["default_headers"],
                         {"anthropic-workspace-id": "workspace-test"})

    def test_missing_anthropic_key_does_not_use_openai_key(self):
        del os.environ["ANTHROPIC_API_KEY"]
        with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
            config.red_openai_client_kwargs()

    def test_blue_stays_locked(self):
        os.environ["ANTHROPIC_MODEL"] = "different-model"
        self.assertEqual(config.get_blue_model(), "lfm3-1.7b")
        self.assertEqual(config.blue_client_kwargs()["api_key"], "fake-liquid")

    def test_setup_does_not_ask_for_openai_key(self):
        del os.environ["OPENAI_API_KEY"]
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")), patch("builtins.print"):
            config.setup_api_key()

    def test_openai_and_gemini_keep_original_routes(self):
        for provider, model in (("openai", "gpt-4o-mini"), ("gemini", "gemini-3.5-flash")):
            with self.subTest(provider=provider):
                os.environ["RED_TEAM_PROVIDER"] = provider
                self.assertEqual(config.get_red_model(), model)
                entry = build_rails_dict()["models"][0]
                self.assertNotEqual(entry["api_key_env_var"], "ANTHROPIC_API_KEY")

    def test_actual_sdk_sends_anthropic_request_with_mock_transport(self):
        # OpenAI SDK 3 uses httpx2; SDK 1/2 uses httpx.
        try:
            import httpx2 as httpx
        except ImportError:
            import httpx
        from openai import OpenAI

        def handler(request):
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer fake-anthropic")
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0,
                "model": "claude-sonnet-4-6",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                             "finish_reason": "stop"}],
            })

        agent, runner = create_openai_pair(name="red", instruction="Banking", app_name="test")
        with OpenAI(**runner.client_kwargs,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            with patch.object(runner, "_client", return_value=client):
                self.assertEqual(asyncio.run(runner.chat(agent, "Hello")), "OK")


if __name__ == "__main__":
    unittest.main()

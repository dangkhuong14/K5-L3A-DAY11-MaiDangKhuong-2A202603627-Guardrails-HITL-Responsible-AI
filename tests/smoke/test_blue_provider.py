"""Blue provider routing checks with fake keys and no network requests."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import config
from core.openai_runtime import create_blue_pair, create_openai_pair


class BlueProviderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_default_keeps_openrouter_lab_model(self):
        os.environ.update(OPENROUTER_API_KEY="fake-router", LIQUID_MODEL="other")
        self.assertEqual(config.get_blue_provider(), "openrouter")
        self.assertEqual(config.get_blue_model(), "liquid/lfm-3-1.7b:free")
        self.assertEqual(config.blue_client_kwargs(), {
            "api_key": "fake-router", "base_url": "https://openrouter.ai/api/v1",
        })

    def test_liquid_uses_own_key_url_and_model(self):
        os.environ.update(
            BLUE_TEAM_PROVIDER=" Liquid ", LIQUID_API_KEY=" fake-liquid ",
            LIQUID_MODEL="lfm3-1.7b", OPENROUTER_API_KEY="fake-router",
            OPENROUTER_BASE_URL="https://router.example/v1",
        )
        agent, runner = create_blue_pair(name="blue", instruction="Banking", app_name="test")
        self.assertEqual(agent.provider, "liquid")
        self.assertEqual(runner.provider, "liquid")
        self.assertEqual(runner.model, "lfm3-1.7b")
        self.assertEqual(runner.client_kwargs, {
            "api_key": "fake-liquid", "base_url": "https://api.liquid.ai/openai/v1",
        })

    def test_liquid_defaults_and_endpoint_override(self):
        os.environ.update(BLUE_TEAM_PROVIDER="liquid", LIQUID_API_KEY="fake-liquid")
        self.assertEqual(config.get_blue_model(), "lfm3-1.7b")
        os.environ.update(LIQUID_MODEL="  ", LIQUID_BASE_URL=" https://liquid.example/v1 ")
        self.assertEqual(config.get_blue_model(), "lfm3-1.7b")
        self.assertEqual(config.blue_client_kwargs()["base_url"], "https://liquid.example/v1")

    def test_missing_liquid_key_does_not_fall_back_to_other_keys(self):
        os.environ.update(
            BLUE_TEAM_PROVIDER="liquid", OPENAI_API_KEY="fake-red",
            OPENROUTER_API_KEY="fake-router",
        )
        with self.assertRaisesRegex(ValueError, "LIQUID_API_KEY"):
            config.blue_client_kwargs()

    def test_unknown_provider_fails_early(self):
        os.environ["BLUE_TEAM_PROVIDER"] = "typo"
        with self.assertRaisesRegex(ValueError, "BLUE_TEAM_PROVIDER"):
            config.get_blue_provider()

    def test_model_is_locked_for_both_providers_despite_env_overrides(self):
        os.environ.update(
            LIQUID_MODEL="lfm2.5-24b-a2b", BLUE_MODEL="other-model",
            OPENROUTER_MODEL="other-model", OPENAI_MODEL="gpt-4o-mini",
        )
        for provider, expected in (
            ("liquid", "lfm3-1.7b"),
            ("openrouter", "liquid/lfm-3-1.7b:free"),
        ):
            with self.subTest(provider=provider):
                os.environ["BLUE_TEAM_PROVIDER"] = provider
                self.assertEqual(config.get_blue_model(), expected)

    def test_setup_does_not_prompt_for_openrouter_when_using_liquid(self):
        os.environ.update(
            BLUE_TEAM_PROVIDER="liquid", LIQUID_API_KEY="fake-liquid",
            RED_TEAM_PROVIDER="openai", OPENAI_API_KEY="fake-red",
        )
        with patch("builtins.input", side_effect=AssertionError("Unexpected key prompt")), \
                patch("builtins.print") as output:
            config.setup_api_key()
        self.assertIn("liquid:lfm3-1.7b", output.call_args_list[0].args[0])
        self.assertIn("[LOCKED]", output.call_args_list[0].args[0])

    def test_red_keeps_its_own_configuration(self):
        os.environ.update(
            BLUE_TEAM_PROVIDER="liquid", LIQUID_API_KEY="fake-liquid",
            LIQUID_MODEL="lfm3-1.7b", RED_TEAM_PROVIDER="openai",
            OPENAI_API_KEY="fake-red", OPENAI_MODEL="gpt-4o-mini",
        )
        _, runner = create_openai_pair(name="red", instruction="Internal", app_name="test")
        self.assertEqual(runner.provider, "openai")
        self.assertEqual(runner.model, "gpt-4o-mini")
        self.assertEqual(runner.client_kwargs, {"api_key": "fake-red"})

    def test_liquid_chat_preserves_input_and_output_hooks(self):
        os.environ.update(BLUE_TEAM_PROVIDER="liquid", LIQUID_API_KEY="fake-liquid")
        agent, runner = create_blue_pair(
            name="blue", instruction="Banking", app_name="test",
            input_hooks=[lambda text: "Blocked" if text == "attack" else None],
            output_hooks=[lambda text: text.replace("sensitive", "[REDACTED]")],
        )
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="sensitive"))],
        )
        with patch.object(runner, "_client", return_value=client) as make_client:
            self.assertEqual(asyncio.run(runner.chat(agent, "attack")), "Blocked")
            make_client.assert_not_called()
            self.assertEqual(asyncio.run(runner.chat(agent, "savings")), "[REDACTED]")
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "lfm3-1.7b")


if __name__ == "__main__":
    unittest.main()

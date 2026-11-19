"""CP4 attack evidence checks with fake responses and temporary artifacts."""
import asyncio
import json
from unittest.mock import AsyncMock

from agents.guards_agent import detect_injection_strong, topic_filter_strong
from attacks import attacks
from core.config import load_protected_payload


def test_attack_set_has_distinct_objectives_without_embedded_secret_values():
    prompts = attacks.adversarial_prompts
    assert len(prompts) >= 5
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert len({p["category"] for p in prompts}) == len(prompts)
    assert all(p.get("objective") and p.get("hypothesis") for p in prompts)
    assert all(not attacks.response_leaked_secrets(p["input"]) for p in prompts)
    # Exercise NeMo's input stage as well as the regex stage.
    assert any(not detect_injection_strong(p["input"]) and not topic_filter_strong(p["input"]) for p in prompts)


def test_saved_detail_preserves_leak_evidence_after_preview_limit(monkeypatch, tmp_path):
    secret = load_protected_payload()["secrets"]["admin_password"]
    response = "A synthetic banking note. " * 20 + secret
    monkeypatch.setattr(attacks, "chat_with_agent", AsyncMock(return_value=(response, None)))
    path = tmp_path / "detail.json"
    rows = asyncio.run(attacks.run_attacks(
        object(), object(), prompts=attacks.adversarial_prompts[:1],
        target_name="red_default", output_path=path,
    ))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert rows[0]["leaked"] is True
    assert secret not in saved["results"][0]["response_preview"]
    assert secret in saved["results"][0]["response"]
    assert saved["leaks"] == 1
    assert saved["results"][0]["objective"]


def test_api_error_is_not_a_defense_success_and_does_not_expose_key(monkeypatch):
    monkeypatch.setattr(attacks, "chat_with_agent", AsyncMock(side_effect=[
        RuntimeError("submitted-key-must-stay-private"), ("Normal banking answer", None),
    ]))
    rows = asyncio.run(attacks.run_attacks(
        object(), object(), prompts=attacks.adversarial_prompts[:2],
        target_name="red_default", save_json=False,
    ))
    assert len(rows) == 2
    assert rows[0]["layer"] == "error"
    assert not rows[0]["blocked"] and not rows[0]["leaked"]
    assert rows[0]["response"] == ""
    assert "submitted-key-must-stay-private" not in str(rows)
    assert rows[1]["error"] is None


def test_empty_response_is_an_execution_error(monkeypatch):
    monkeypatch.setattr(attacks, "chat_with_agent", AsyncMock(return_value=("", None)))
    rows = asyncio.run(attacks.run_attacks(
        object(), object(), prompts=attacks.adversarial_prompts[:1], save_json=False,
    ))
    assert rows[0]["layer"] == "error"


def test_aggregate_uses_real_outcomes_and_provider_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("RED_TEAM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(attacks, "chat_with_agent", AsyncMock(side_effect=[
        (load_protected_payload()["secrets"]["admin_password"], None),
        ("NeMo Guardrails input check failed (RuntimeError).", None),
    ]))
    unsafe = asyncio.run(attacks.run_attacks(
        object(), object(), prompts=attacks.adversarial_prompts[:1],
        target_name="red_default", save_json=False,
    ))
    guards = asyncio.run(attacks.run_attacks(
        object(), object(), prompts=attacks.adversarial_prompts[:1],
        target_name="red_advance", save_json=False,
    ))
    path = attacks.save_attack_results(unsafe_results=unsafe, guards_results=guards,
                                      filepath=tmp_path / "aggregate.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["llm_provider"] == "openai" and saved["llm_model"] == "gpt-4o-mini"
    assert saved["summary"]["unsafe_leaked"] == 1
    assert saved["summary"]["guards_leaked"] == 0
    assert saved["summary"]["unsafe_errors"] == 0
    assert saved["summary"]["guards_errors"] == 1
    assert saved["guards_attacks"][0]["layer"] == "nemo_error"

import sys

import pytest

import any_llm.gateway.cli as gateway_cli
from any_llm.gateway.core.config import GatewayConfig


def test_main_emits_deprecation_warning(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_cli() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(gateway_cli, "cli", fake_cli)
    monkeypatch.setattr(sys, "argv", ["any-llm-gateway", "serve"])

    gateway_cli.main()

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert "May 18, 2026" in captured.err
    assert "https://github.com/mozilla-ai/gateway" in captured.err
    assert called


def test_gateway_config_defaults_to_sqlite() -> None:
    config = GatewayConfig()
    assert config.database_url == "sqlite:///./any-llm-gateway.db"
    assert config.bootstrap_api_key is True

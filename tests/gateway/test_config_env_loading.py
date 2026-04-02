import os

from any_llm.gateway.core.config import load_config


def test_load_config_loads_provider_env_vars_from_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\nGATEWAY_MASTER_KEY=gateway-master\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_MASTER_KEY", raising=False)

    config = load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "from-dotenv"
    assert config.master_key == "gateway-master"


def test_load_config_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")

    load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "already-set"

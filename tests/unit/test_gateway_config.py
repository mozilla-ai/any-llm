"""Tests for gateway configuration with environment variables."""

import json
import os
from tempfile import NamedTemporaryFile

import pytest

from any_llm.gateway.config import GatewayConfig, load_config


class TestGatewayConfigEnvVars:
    """Test that all config parameters can be set via environment variables."""

    def test_simple_string_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test simple string configuration via environment variables."""
        monkeypatch.setenv("GATEWAY_HOST", "127.0.0.1")
        monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://test:test@localhost/test")
        monkeypatch.setenv("GATEWAY_MASTER_KEY", "test-master-key")

        config = GatewayConfig()

        assert config.host == "127.0.0.1"
        assert config.database_url == "postgresql://test:test@localhost/test"
        assert config.master_key == "test-master-key"

    def test_integer_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test integer configuration via environment variables."""
        monkeypatch.setenv("GATEWAY_PORT", "9000")

        config = GatewayConfig()

        assert config.port == 9000

    def test_boolean_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test boolean configuration via environment variables."""
        monkeypatch.setenv("GATEWAY_AUTO_MIGRATE", "false")

        config = GatewayConfig()

        assert config.auto_migrate is False

        monkeypatch.setenv("GATEWAY_AUTO_MIGRATE", "true")
        config = GatewayConfig()

        assert config.auto_migrate is True

    def test_providers_json_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test providers configuration via JSON environment variable."""
        providers_json = json.dumps({
            "openai": {
                "api_key": "sk-test-key",
                "api_base": "https://api.openai.com/v1",
            },
            "anthropic": {
                "api_key": "sk-ant-test",
            },
        })
        monkeypatch.setenv("GATEWAY_PROVIDERS", providers_json)

        config = GatewayConfig()

        assert "openai" in config.providers
        assert config.providers["openai"]["api_key"] == "sk-test-key"
        assert config.providers["openai"]["api_base"] == "https://api.openai.com/v1"
        assert "anthropic" in config.providers
        assert config.providers["anthropic"]["api_key"] == "sk-ant-test"

    def test_pricing_json_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test pricing configuration via JSON environment variable."""
        pricing_json = json.dumps({
            "openai:gpt-4": {
                "input_price_per_million": 30.0,
                "output_price_per_million": 60.0,
            },
            "anthropic:claude-3-opus": {
                "input_price_per_million": 15.0,
                "output_price_per_million": 75.0,
            },
        })
        monkeypatch.setenv("GATEWAY_PRICING", pricing_json)

        config = GatewayConfig()

        assert "openai:gpt-4" in config.pricing
        assert config.pricing["openai:gpt-4"].input_price_per_million == 30.0
        assert config.pricing["openai:gpt-4"].output_price_per_million == 60.0
        assert "anthropic:claude-3-opus" in config.pricing
        assert config.pricing["anthropic:claude-3-opus"].input_price_per_million == 15.0
        assert config.pricing["anthropic:claude-3-opus"].output_price_per_million == 75.0

    def test_invalid_providers_json_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid JSON in GATEWAY_PROVIDERS raises an error."""
        from pydantic_settings import SettingsError

        monkeypatch.setenv("GATEWAY_PROVIDERS", "{invalid json")

        with pytest.raises(SettingsError, match='error parsing value for field "providers"'):
            GatewayConfig()

    def test_invalid_pricing_json_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid JSON in GATEWAY_PRICING raises an error."""
        from pydantic_settings import SettingsError

        monkeypatch.setenv("GATEWAY_PRICING", "{invalid json")

        with pytest.raises(SettingsError, match='error parsing value for field "pricing"'):
            GatewayConfig()

    def test_providers_as_list_becomes_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that array JSON in GATEWAY_PROVIDERS is converted to empty dict."""
        monkeypatch.setenv("GATEWAY_PROVIDERS", '["array", "not", "object"]')

        config = GatewayConfig()

        # Pydantic-settings will parse the JSON array, but our validator will return empty dict
        assert config.providers == {}

    def test_pricing_as_list_becomes_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that array JSON in GATEWAY_PRICING is converted to empty dict."""
        monkeypatch.setenv("GATEWAY_PRICING", '["array", "not", "object"]')

        config = GatewayConfig()

        # Pydantic-settings will parse the JSON array, but our validator will return empty dict
        assert config.pricing == {}

    def test_all_config_params_via_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that all configuration parameters can be set via environment variables."""
        monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://env:env@localhost/env")
        monkeypatch.setenv("GATEWAY_AUTO_MIGRATE", "false")
        monkeypatch.setenv("GATEWAY_HOST", "192.168.1.1")
        monkeypatch.setenv("GATEWAY_PORT", "7000")
        monkeypatch.setenv("GATEWAY_MASTER_KEY", "env-master-key")
        monkeypatch.setenv("GATEWAY_PROVIDERS", '{"test": {"key": "value"}}')
        monkeypatch.setenv(
            "GATEWAY_PRICING", '{"test:model": {"input_price_per_million": 1.0, "output_price_per_million": 2.0}}'
        )

        config = GatewayConfig()

        assert config.database_url == "postgresql://env:env@localhost/env"
        assert config.auto_migrate is False
        assert config.host == "192.168.1.1"
        assert config.port == 7000
        assert config.master_key == "env-master-key"
        assert config.providers == {"test": {"key": "value"}}
        assert "test:model" in config.pricing
        assert config.pricing["test:model"].input_price_per_million == 1.0


class TestLoadConfigPrecedence:
    """Test that environment variables take precedence over YAML config."""

    def test_env_vars_override_yaml_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override YAML configuration."""
        yaml_content = """
database_url: "postgresql://yaml:yaml@localhost/yaml"
host: "0.0.0.0"
port: 8000
master_key: "yaml-master-key"
providers:
  openai:
    api_key: "yaml-openai-key"
pricing:
  openai:gpt-4:
    input_price_per_million: 10.0
    output_price_per_million: 20.0
"""

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            # Set environment variables that should override YAML
            monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://env:env@localhost/env")
            monkeypatch.setenv("GATEWAY_PORT", "9000")
            monkeypatch.setenv("GATEWAY_MASTER_KEY", "env-master-key")
            monkeypatch.setenv("GATEWAY_PROVIDERS", '{"anthropic": {"api_key": "env-anthropic-key"}}')

            config = load_config(config_path)

            # Environment variables should take precedence
            assert config.database_url == "postgresql://env:env@localhost/env"
            assert config.port == 9000
            assert config.master_key == "env-master-key"
            assert "anthropic" in config.providers
            assert config.providers["anthropic"]["api_key"] == "env-anthropic-key"

            # YAML value should be used when no env var is set
            assert config.host == "0.0.0.0"  # noqa: S104
        finally:
            os.unlink(config_path)

    def test_load_config_without_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that load_config works without a config file (env vars only)."""
        monkeypatch.setenv("GATEWAY_HOST", "10.0.0.1")
        monkeypatch.setenv("GATEWAY_PORT", "5000")
        monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://test:test@localhost/test")
        monkeypatch.setenv("GATEWAY_MASTER_KEY", "test-key")

        config = load_config(None)

        assert config.host == "10.0.0.1"
        assert config.port == 5000
        assert config.database_url == "postgresql://test:test@localhost/test"
        assert config.master_key == "test-key"

    def test_defaults_used_when_no_config_or_env(self) -> None:
        """Test that default values are used when no config file or env vars are set."""
        config = GatewayConfig()

        assert config.host == "0.0.0.0"  # noqa: S104
        assert config.port == 8000
        assert config.database_url == "postgresql://postgres:postgres@localhost:5432/any_llm_gateway"
        assert config.auto_migrate is True
        assert config.master_key is None
        assert config.providers == {}
        assert config.pricing == {}

    def test_yaml_env_var_substitution_still_works(self) -> None:
        """Test that ${VAR} substitution in YAML still works."""
        yaml_content = """
database_url: "postgresql://postgres:postgres@localhost/db"
master_key: "${GATEWAY_MASTER_KEY}"
"""

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            os.environ["GATEWAY_MASTER_KEY"] = "substituted-key"

            config = load_config(config_path)

            assert config.master_key == "substituted-key"
        finally:
            os.unlink(config_path)
            if "GATEWAY_MASTER_KEY" in os.environ:
                del os.environ["GATEWAY_MASTER_KEY"]

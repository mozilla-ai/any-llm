import sys
from unittest.mock import patch

import pytest

from any_llm.provider import ApiConfig, ProviderFactory


def test_provider_with_no_packages_installed() -> None:
    with patch.dict(sys.modules, dict.fromkeys(["fireworks"])):
        try:
            import any_llm.providers.fireworks  # noqa: F401
        except ImportError:
            pytest.fail("Import raised an unexpected ImportError")


def test_call_to_provider_with_no_packages_installed() -> None:
    packages = ["instructor", "fireworks"]
    with patch.dict(sys.modules, dict.fromkeys(packages)):
        with pytest.raises(ImportError, match="fireworks required packages are not installed"):
            ProviderFactory.create_provider("fireworks", ApiConfig())

import sys
from unittest.mock import patch

import pytest

from any_llm.provider import ApiConfig, ProviderFactory


def test_provider_with_no_packages_installed() -> None:
    with patch.dict(sys.modules, dict.fromkeys(["xai_sdk"])):
        try:
            import any_llm.providers.xai  # noqa: F401
        except ImportError:
            pytest.fail("Import raised an unexpected ImportError")


def test_call_to_provider_with_no_packages_installed() -> None:
    packages = ["xai_sdk"]
    with patch.dict(sys.modules, dict.fromkeys(packages)):
        with pytest.raises(ImportError, match="xai required packages are not installed"):
            ProviderFactory.create_provider("xai", ApiConfig())

import tomllib
from pathlib import Path

from any_llm.provider import ProviderName


def test_all_providers_have_pyproject_options() -> None:
    """Test that all providers have corresponding optional dependencies in pyproject.toml."""

    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    optional_deps = set(pyproject_data["project"]["optional-dependencies"].keys())

    all_providers = {provider.value for provider in ProviderName}

    missing_deps = all_providers - optional_deps

    assert not missing_deps, f"Missing optional dependencies for providers: {missing_deps}"

    all_group_str = pyproject_data["project"]["optional-dependencies"]["all"][0]
    providers_in_all = set()
    if "[" in all_group_str and "]" in all_group_str:
        providers_part = all_group_str.split("[")[1].split("]")[0]
        providers_in_all = {dep.strip() for dep in providers_part.split(",")}

    missing_from_all = all_providers - providers_in_all
    assert not missing_from_all, f"Missing providers in 'all' optional dependency group: {missing_from_all}"

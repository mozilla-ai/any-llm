#!/usr/bin/env python3
"""
Script to check which providers are being skipped due to missing API keys.

This script attempts to instantiate each provider and reports which ones
fail due to missing API keys, missing packages, or other issues.
"""

import os
import sys
from pathlib import Path

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def check_provider_status():
    """Check the status of all providers and categorize them."""
    available_providers = []
    missing_api_keys = []
    missing_packages = []
    other_errors = []

    print("Checking provider status...")
    print("=" * 50)

    for provider_name in LLMProvider:
        try:
            provider_class = AnyLLM.get_provider_class(provider_name)

            if provider_class.MISSING_PACKAGES_ERROR is not None:
                missing_packages.append(
                    {
                        "name": provider_name.value,
                        "error": str(provider_class.MISSING_PACKAGES_ERROR),
                        "env_var": provider_class.ENV_API_KEY_NAME,
                    }
                )
                continue

            available_providers.append(
                {
                    "name": provider_name.value,
                    "env_var": provider_class.ENV_API_KEY_NAME,
                    "api_key_set": bool(os.getenv(provider_class.ENV_API_KEY_NAME)),
                }
            )

        except MissingApiKeyError as e:
            missing_api_keys.append({"name": provider_name.value, "env_var": e.env_var_name, "error": str(e)})
        except ImportError as e:
            missing_packages.append({"name": provider_name.value, "error": str(e), "env_var": "N/A"})
        except Exception as e:
            other_errors.append({"name": provider_name.value, "error": str(e), "error_type": type(e).__name__})

    return available_providers, missing_api_keys, missing_packages, other_errors


def print_results(available_providers, missing_api_keys, missing_packages, other_errors):
    """Print formatted results of the provider status check."""

    if available_providers:
        print(f"✅ Available Providers ({len(available_providers)}):")
        for provider in sorted(available_providers, key=lambda x: x["name"]):
            key_status = "🔑" if provider["api_key_set"] else "🔓"
            print(f"  {key_status} {provider['name']} (env: {provider['env_var']})")
        print()

    if missing_api_keys:
        print(f"🔑 Missing API Keys ({len(missing_api_keys)}):")
        for provider in sorted(missing_api_keys, key=lambda x: x["name"]):
            print(f"  ❌ {provider['name']} - Set {provider['env_var']}")
        print()

    if missing_packages:
        print(f"📦 Missing Packages ({len(missing_packages)}):")
        for provider in sorted(missing_packages, key=lambda x: x["name"]):
            print(f"  ❌ {provider['name']} - {provider['error']}")
        print()

    if other_errors:
        print(f"⚠️  Other Errors ({len(other_errors)}):")
        for provider in sorted(other_errors, key=lambda x: x["name"]):
            print(f"  ❌ {provider['name']} ({provider['error_type']}) - {provider['error']}")
        print()

    total_providers = len(list(LLMProvider))
    available_count = len(available_providers)
    missing_keys_count = len(missing_api_keys)
    missing_packages_count = len(missing_packages)
    other_errors_count = len(other_errors)

    print("📊 Summary:")
    print(f"  Total providers: {total_providers}")
    print(f"  Available: {available_count}")
    print(f"  Missing API keys: {missing_keys_count}")
    print(f"  Missing packages: {missing_packages_count}")
    print(f"  Other errors: {other_errors_count}")

    if missing_api_keys:
        print("\n💡 To fix missing API keys, set these environment variables:")
        for provider in sorted(missing_api_keys, key=lambda x: x["name"]):
            print(f"  export {provider['env_var']}='your-api-key-here'")


def main():
    """Main function to run the provider status check."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        print("\nUsage:")
        print("  python scripts/check_missing_api_keys.py")
        print("  python scripts/check_missing_api_keys.py --help")
        return

    try:
        available, missing_keys, missing_packages, other_errors = check_provider_status()
        print_results(available, missing_keys, missing_packages, other_errors)
    except Exception as e:
        print(f"Error running provider check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

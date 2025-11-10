#!/usr/bin/env python3
"""Generate OpenAPI specification for the any-llm-gateway.

This script creates the FastAPI application and exports its OpenAPI specification
to a JSON file. It can be run in two modes:
- Generate mode (default): Writes the spec to docs/openapi.json
- Check mode (--check): Compares generated spec with existing file and exits with
  error if they differ (useful for CI/CD)
"""

import argparse
import json
import sys
from pathlib import Path

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.server import create_app


def _add_security_schemes(spec: dict) -> dict:
    """Add security scheme definitions to OpenAPI spec.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        Modified specification with security schemes

    """
    # Ensure components section exists
    if "components" not in spec:
        spec["components"] = {}

    # Add security schemes
    spec["components"]["securitySchemes"] = {
        "APIKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER,
            "description": (
                "API key authentication for chat completions and user operations. "
                "Format: `Bearer <your-api-key>`"
            ),
        },
        "MasterKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER,
            "description": (
                "Master key authentication for administrative operations "
                "(key management, user management, budget configuration, pricing configuration). "
                "Format: `Bearer <your-master-key>`"
            ),
        },
    }

    # Add security requirements to endpoints based on their tags and dependencies
    if "paths" in spec:
        for path, methods in spec["paths"].items():
            for method, endpoint in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    # Determine security based on tags and common patterns
                    tags = endpoint.get("tags", [])

                    # Health endpoint - no auth required
                    if "health" in tags:
                        continue

                    # Get existing description
                    existing_desc = endpoint.get("description", "")

                    # Admin endpoints (keys, users, budgets, pricing) require master key
                    if any(tag in ["keys", "users", "budgets", "pricing"] for tag in tags):
                        endpoint["security"] = [{"MasterKeyAuth": []}]
                        # Add security note to description
                        security_note = (
                            "\n\n**Authentication:** Requires Master Key\n\n"
                            "Include header: `X-AnyLLM-Key: Bearer <your-master-key>`"
                        )
                        endpoint["description"] = existing_desc + security_note

                    # Chat endpoints can use either API key or master key
                    elif "chat" in tags:
                        endpoint["security"] = [{"APIKeyAuth": []}, {"MasterKeyAuth": []}]
                        # Add security note to description
                        security_note = (
                            "\n\n**Authentication:** Requires API Key or Master Key\n\n"
                            "Include header: `X-AnyLLM-Key: Bearer <your-api-key>` or "
                            "`X-AnyLLM-Key: Bearer <your-master-key>`"
                        )
                        endpoint["description"] = existing_desc + security_note

                    # Default: require API key
                    else:
                        endpoint["security"] = [{"APIKeyAuth": []}]
                        # Add security note to description
                        security_note = (
                            "\n\n**Authentication:** Requires API Key\n\n"
                            "Include header: `X-AnyLLM-Key: Bearer <your-api-key>`"
                        )
                        endpoint["description"] = existing_desc + security_note

    return spec


def generate_openapi_spec() -> dict:
    """Generate OpenAPI specification from FastAPI app.

    Returns:
        OpenAPI specification as a dictionary

    """
    config = GatewayConfig(database_url="sqlite:///:memory:")
    app = create_app(config)
    spec = app.openapi()

    # Add security schemes to the OpenAPI spec
    spec = _add_security_schemes(spec)

    return spec


def write_spec(spec: dict, output_path: Path) -> None:
    """Write OpenAPI spec to file with pretty formatting.

    Args:
        spec: OpenAPI specification dictionary
        output_path: Path to output JSON file

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
        f.write("\n")


def check_spec(spec: dict, existing_path: Path) -> bool:
    """Check if generated spec matches existing file.

    Args:
        spec: Generated OpenAPI specification
        existing_path: Path to existing spec file

    Returns:
        True if specs match, False otherwise

    """
    if not existing_path.exists():
        print(f"Error: {existing_path} does not exist", file=sys.stderr)
        return False

    with open(existing_path, encoding="utf-8") as f:
        existing_spec = json.load(f)

    # Create copies to avoid modifying originals
    spec_copy = spec.copy()
    existing_copy = existing_spec.copy()

    # Remove version from comparison since it's dynamically generated from git
    if "info" in spec_copy and "version" in spec_copy["info"]:
        spec_copy["info"] = spec_copy["info"].copy()
        spec_copy["info"].pop("version")
    if "info" in existing_copy and "version" in existing_copy["info"]:
        existing_copy["info"] = existing_copy["info"].copy()
        existing_copy["info"].pop("version")

    generated_json = json.dumps(spec_copy, indent=2, sort_keys=True)
    existing_json = json.dumps(existing_copy, indent=2, sort_keys=True)
    if generated_json != existing_json:
        print("Generated spec does not match existing spec", file=sys.stderr)
        print("Generated spec:")
        print(generated_json)
        print("Existing spec:")
        print(existing_json)
        return False

    return generated_json == existing_json


def main() -> int:
    """Generate or check OpenAPI specification.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    parser = argparse.ArgumentParser(description="Generate OpenAPI specification")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated spec matches existing file (for CI/CD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "gateway" / "openapi.json",
        help="Output path for OpenAPI spec (default: docs/gateway/openapi.json)",
    )

    args = parser.parse_args()

    print("Generating OpenAPI specification...")
    spec = generate_openapi_spec()

    if args.check:
        print(f"Checking if {args.output} is up to date...")
        if check_spec(spec, args.output):
            print("✓ OpenAPI spec is up to date")
            return 0
        print("✗ OpenAPI spec is out of date", file=sys.stderr)
        print("Run 'python scripts/generate_openapi.py' to update it", file=sys.stderr)
        return 1

    write_spec(spec, args.output)
    print(f"✓ OpenAPI spec written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

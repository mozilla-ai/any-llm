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

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.server import create_app


def generate_openapi_spec() -> dict:
    """Generate OpenAPI specification from FastAPI app.

    Returns:
        OpenAPI specification as a dictionary

    """
    config = GatewayConfig(database_url="sqlite:///:memory:")
    app = create_app(config)
    return app.openapi()


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

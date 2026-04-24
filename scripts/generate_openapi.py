#!/usr/bin/env python3
"""Generate OpenAPI specification for the any-llm-gateway.

This script creates the FastAPI application and exports its OpenAPI specification
to a JSON file. Writes the spec to docs/openapi.json by default.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.main import create_app


def generate_openapi_spec() -> dict:
    """Generate OpenAPI specification from FastAPI app.

    Returns:
        OpenAPI specification as a dictionary

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        database_url = f"sqlite:///{Path(tmpdir) / 'openapi.db'}"
        config = GatewayConfig(database_url=database_url, bootstrap_api_key=False)
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


def main() -> int:
    """Generate OpenAPI specification.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    parser = argparse.ArgumentParser(description="Generate OpenAPI specification")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "openapi.json",
        help="Output path for OpenAPI spec (default: docs/openapi.json)",
    )

    args = parser.parse_args()

    print("Generating OpenAPI specification...")
    spec = generate_openapi_spec()
    write_spec(spec, args.output)
    print(f"✓ OpenAPI spec written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

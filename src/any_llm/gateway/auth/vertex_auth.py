"""Helper module for loading Google Cloud Vertex AI credentials."""

import atexit
import json
import os
import tempfile
from typing import Any

from fastapi import HTTPException, status

_temp_credential_file: str | None = None


def _cleanup_temp_credentials() -> None:
    """Remove temporary credential file on process exit."""
    if _temp_credential_file and os.path.exists(_temp_credential_file):
        try:
            os.remove(_temp_credential_file)
        except OSError:
            pass


atexit.register(_cleanup_temp_credentials)


def setup_vertex_environment(
    credentials: str | dict[str, Any] | None = None,
    project: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Vertex AI environment variables setup for any-llm.

    The google.genai.Client used by any-llm reads credentials from environment variables.
    This function sets up GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT,
    and GOOGLE_CLOUD_LOCATION as needed.

    Args:
        credentials: Path to service account JSON file, JSON string, or dict
        project: Optional GCP project ID (extracted from credentials if not provided)
        location: Optional GCP location (default: "us-central1")

    Returns:
        Empty dict (environment is configured via env vars)

    Raises:
        HTTPException: If credentials cannot be loaded or project cannot be determined

    """
    global _temp_credential_file  # noqa: PLW0603

    env_updates: dict[str, str] = {}

    if credentials is not None:
        json_obj = None

        if isinstance(credentials, str):
            try:
                if os.path.exists(credentials):
                    with open(credentials, encoding="utf-8") as f:
                        json_obj = json.load(f)
                    env_updates["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
                else:
                    json_obj = json.loads(credentials)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unable to load vertex credentials: {e}",
                ) from e
        elif isinstance(credentials, dict):
            json_obj = credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid credentials type: {type(credentials)}",
            )

        if json_obj and "GOOGLE_APPLICATION_CREDENTIALS" not in env_updates:
            if _temp_credential_file and os.path.exists(_temp_credential_file):
                env_updates["GOOGLE_APPLICATION_CREDENTIALS"] = _temp_credential_file
            else:
                temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump(json_obj, temp_file)
                temp_file.close()
                _temp_credential_file = temp_file.name
                env_updates["GOOGLE_APPLICATION_CREDENTIALS"] = _temp_credential_file

        if project is None and json_obj:
            project = json_obj.get("project_id")

    if project:
        env_updates["GOOGLE_CLOUD_PROJECT"] = project
    elif "GOOGLE_CLOUD_PROJECT" not in os.environ:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not resolve GCP project ID from credentials or configuration",
        )

    if location:
        env_updates["GOOGLE_CLOUD_LOCATION"] = location
    elif "GOOGLE_CLOUD_LOCATION" not in os.environ:
        env_updates["GOOGLE_CLOUD_LOCATION"] = "us-central1"

    for key, value in env_updates.items():
        os.environ[key] = value

    return {}

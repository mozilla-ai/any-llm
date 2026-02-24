"""Tests for Vertex AI credential file caching and cleanup."""

import json
import os

from any_llm.gateway.auth import vertex_auth
from any_llm.gateway.auth.vertex_auth import setup_vertex_environment


def test_temp_credential_file_reused_across_calls() -> None:
    """Test that repeated calls reuse the same temp credential file."""
    # Reset cached file
    original = vertex_auth._temp_credential_file
    vertex_auth._temp_credential_file = None

    creds = {"type": "service_account", "project_id": "test-project"}

    try:
        setup_vertex_environment(credentials=creds, project="test-project")
        first_file = vertex_auth._temp_credential_file
        assert first_file is not None
        assert os.path.exists(first_file)

        setup_vertex_environment(credentials=creds, project="test-project")
        second_file = vertex_auth._temp_credential_file

        assert first_file == second_file, "Should reuse the same temp file"
    finally:
        # Clean up
        if vertex_auth._temp_credential_file and os.path.exists(vertex_auth._temp_credential_file):
            os.remove(vertex_auth._temp_credential_file)
        vertex_auth._temp_credential_file = original
        # Clean up env vars we set
        for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]:
            os.environ.pop(var, None)


def test_temp_credential_file_contains_valid_json() -> None:
    """Test that the temp credential file contains valid JSON."""
    original = vertex_auth._temp_credential_file
    vertex_auth._temp_credential_file = None

    creds = {
        "type": "service_account",
        "project_id": "test-project",
        "client_email": "test@test.iam.gserviceaccount.com",
    }

    try:
        setup_vertex_environment(credentials=creds, project="test-project")
        temp_file = vertex_auth._temp_credential_file
        assert temp_file is not None

        with open(temp_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["project_id"] == "test-project"
    finally:
        if vertex_auth._temp_credential_file and os.path.exists(vertex_auth._temp_credential_file):
            os.remove(vertex_auth._temp_credential_file)
        vertex_auth._temp_credential_file = original
        for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]:
            os.environ.pop(var, None)

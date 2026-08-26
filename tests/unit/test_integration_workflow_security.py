from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"


def test_secret_bearing_integration_workflow_never_checks_out_pr_head() -> None:
    workflow = WORKFLOW.read_text()

    assert "pull_request_target:" in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "allow-unsafe-pr-checkout" not in workflow
    assert workflow.count("github.event.pull_request.base.sha || github.sha") == 3

import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"


def test_secret_bearing_integration_workflow_never_checks_out_pr_head() -> None:
    workflow = WORKFLOW.read_text()

    assert "pull_request_target:" in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "allow-unsafe-pr-checkout" not in workflow
    checkout_steps = re.findall(
        r"(?m)^[ \t]*- uses:\s*actions/checkout@[^\n]+\n[ \t]+with:\n[ \t]+ref:\s*([^\r\n]+)",
        workflow,
    )
    assert len(checkout_steps) == 3
    assert all(ref == "${{ github.event.pull_request.base.sha || github.sha }}" for ref in checkout_steps)
    checkout_commits = re.findall(r"actions/checkout@([^\s]+)", workflow)
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in checkout_commits)

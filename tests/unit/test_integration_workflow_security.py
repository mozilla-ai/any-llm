import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"


def test_secret_bearing_integration_workflow_never_checks_out_pr_head() -> None:
    """Require every secret-bearing checkout step to use the trusted revision."""
    workflow = WORKFLOW.read_text()

    assert "pull_request_target:" in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "allow-unsafe-pr-checkout" not in workflow
    checkout_steps = re.findall(
        r"(?ms)^(?P<indent>[ \t]*)- uses:\s*actions/checkout@(?P<commit>[^\s]+)\s*\n"
        r"(?P<body>.*?)(?=^(?P=indent)- |\Z)",
        workflow,
    )
    checkout_commits = re.findall(r"actions/checkout@([^\s]+)", workflow)
    assert len(checkout_steps) == len(checkout_commits) == 3
    trusted_ref = "${{ github.event.pull_request.base.sha || github.sha }}"
    for _, _, body in checkout_steps:
        ref = re.search(r"(?m)^[ \t]+ref:\s*([^\r\n]+)", body)
        assert ref is not None
        assert ref.group(1) == trusted_ref
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in checkout_commits)

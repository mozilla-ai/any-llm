import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"


def test_labeled_integration_workflow_checks_out_reviewed_pr_head() -> None:
    """Require labeled integration jobs to test the PR while preserving checkout safeguards."""
    workflow = WORKFLOW.read_text()

    assert "pull_request_target:" in workflow
    assert "types: [labeled]" in workflow
    assert "github.event.label.name == 'run-integration-tests'" in workflow
    checkout_steps = re.findall(
        r"(?ms)^(?P<indent>[ \t]*)- uses:\s*actions/checkout@(?P<commit>[^\s]+)\s*\n"
        r"(?P<body>.*?)(?=^(?P=indent)- |\Z)",
        workflow,
    )
    checkout_commits = re.findall(r"actions/checkout@([^\s]+)", workflow)
    assert len(checkout_steps) == len(checkout_commits) == 3
    base_ref = "${{ github.event.pull_request.base.sha || github.sha }}"
    pr_head_ref = "${{ github.event.pull_request.head.sha || github.sha }}"
    refs = []
    for _, _, body in checkout_steps:
        ref = re.search(r"(?m)^[ \t]+ref:\s*([^\r\n]+)", body)
        assert ref is not None
        refs.append(ref.group(1))
        assert "persist-credentials: false" in body
    assert refs == [base_ref, pr_head_ref, pr_head_ref]
    assert "allow-unsafe-pr-checkout: true" not in checkout_steps[0][2]
    assert all("allow-unsafe-pr-checkout: true" in body for _, _, body in checkout_steps[1:])
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in checkout_commits)

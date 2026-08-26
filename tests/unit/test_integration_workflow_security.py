import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"


def test_secret_bearing_integration_workflow_never_checks_out_pr_head() -> None:
    workflow = WORKFLOW.read_text()

    assert "pull_request_target:" in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "allow-unsafe-pr-checkout" not in workflow
    assert workflow.count("github.event.pull_request.base.sha || github.sha") == 3

    checkout_refs = re.findall(r"uses:\s*actions/checkout@([^\s]+)", workflow)
    assert checkout_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in checkout_refs)

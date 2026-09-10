import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "tests-integration.yaml"
JOB_PATTERN = re.compile(
    r"(?ms)^  (?P<name>[A-Za-z0-9-]+):\n"
    r"(?P<body>.*?)(?=^  [A-Za-z0-9-]+:\n|\Z)"
)


def test_labeled_integration_workflow_checks_out_reviewed_pr_head() -> None:
    """Require labeled integration jobs to test the PR while preserving checkout safeguards."""
    workflow = WORKFLOW.read_text()
    jobs = {match.group("name"): match.group("body") for match in JOB_PATTERN.finditer(workflow)}

    assert "pull_request_target:" in workflow
    assert "types: [labeled]" in workflow
    authorization_job = jobs["authorize-label"]
    assert "collaborators/${LABEL_ACTOR}/permission" in authorization_job
    assert "LABEL_ACTOR: ${{ github.event.sender.login }}" in authorization_job
    assert "admin|write" in authorization_job
    assert "*)" in authorization_job
    assert 'echo "approved=false"' in authorization_job
    expected_providers_job = jobs["expected-providers"]
    assert "needs: authorize-label" in expected_providers_job
    assert "needs.authorize-label.outputs.approved == 'true'" in expected_providers_job
    assert "github.event_name != 'pull_request_target'" in expected_providers_job
    assert "github.event.label.name == 'run-integration-tests'" in expected_providers_job
    for job_name in ("run-integration-tests", "run-local-integration-tests"):
        assert "needs: [authorize-label, expected-providers, determine-jobs-to-run]" in jobs[job_name]

    checkout_steps = re.findall(
        r"(?ms)^(?P<indent>[ \t]*)- uses:\s*actions/checkout@(?P<commit>[^\s]+)\s*\n"
        r"(?P<body>.*?)(?=^(?P=indent)- |\Z)",
        workflow,
    )
    checkout_commits = re.findall(r"actions/checkout@([^\s]+)", workflow)
    assert len(checkout_steps) == len(checkout_commits) == 3
    base_ref = "${{ github.event.pull_request.base.sha || github.sha }}"
    pr_head_ref = "${{ github.event.pull_request.head.sha || github.sha }}"
    assert f"ref: {base_ref}" in jobs["determine-jobs-to-run"]
    assert f"ref: {pr_head_ref}" in jobs["run-integration-tests"]
    assert f"ref: {pr_head_ref}" in jobs["run-local-integration-tests"]
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

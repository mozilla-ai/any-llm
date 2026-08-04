---
name: integration-test-triage
description: Triage a red any-llm integration test suite. Use when integration tests are failing, when deciding whether a failure is a real regression, or before skipping or "fixing" an integration test.
---

# Integration test triage

Integration tests run on `main` post-merge, and on a PR only via the `run-integration-tests`
label (auto-removed after each run). They are not a merge gate, so failures accumulate.

A red suite is usually a backlog of unrelated causes, not one regression. Build a test's
history before assuming a recent break: list the recent failed runs
(`gh run list --workflow "Integration Tests" --status failure`), then read the failing steps
of the ones you need (`gh run view <id> --log-failed`). Root-cause each failure as one of:

- **An any-llm bug**: fix it and add a test.
- **A deprecated or wrong test model**: update `tests/conftest.py` and cite the provider's docs.
- **Missing or invalid credentials**: a 401/403 from one provider means its key is absent or
  expired. Name the secret; never turn it into a code change.
- **Missing CI infrastructure**: skip it, naming the missing piece in the skip reason.
- **Rate limiting or quota exhaustion**: 429s and quota errors. The suite runs under `-n auto`,
  so re-run serially to tell contention apart from an exhausted account.
- **A provider outage**: errors across every test for one provider. Confirm against that
  provider's public status page before calling it transient.

Never record a cause as "flaky" or "does not reliably support". Every skip and fix needs a
concrete reason (HTTP status, deprecated model, missing CI infra).

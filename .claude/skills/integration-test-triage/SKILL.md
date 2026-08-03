---
name: integration-test-triage
description: Triage a red any-llm integration test suite. Use when integration tests are failing, when deciding whether a failure is a real regression, or before skipping or "fixing" an integration test.
---

# Integration test triage

Integration tests run on `main` post-merge, and on a PR only via the `run-integration-tests`
label (auto-removed after each run). They are not a merge gate, so failures accumulate.

A red suite is usually a backlog of unrelated causes, not one regression. Check a test's
history (`gh run view <id> --log`) before assuming a recent break, then root-cause each
failure as one of:

- **An any-llm bug**: fix it and add a test.
- **A deprecated or wrong test model**: update `tests/conftest.py` and cite the provider's docs.
- **Missing CI infrastructure**: skip it, naming the missing piece in the skip reason.
- **A provider outage**: a whole provider failing at once (e.g. 429s across every test) is transient.

Never record a cause as "flaky" or "does not reliably support" — every skip and fix needs a
concrete reason (HTTP status, deprecated model, missing CI infra).

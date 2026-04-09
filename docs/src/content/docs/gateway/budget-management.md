---
title: Budget Management
description: Configure spending limits, budget tiers, and automatic resets
---

Budgets provide shared spending limits that can be assigned to multiple users. This allows you to create budget tiers (like "Free", "Pro", "Enterprise") and enforce spending limits across groups of users.

## Creating a Budget

```bash
# Create a budget with a $10.00 spending limit and monthly resets (30 days = 2592000 seconds)
curl -X POST http://localhost:8000/v1/budgets \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "max_budget": 10.0,
    "budget_duration_sec": 2592000
  }'
```

<details>
<summary>Sample Response</summary>

```json
{
  "budget_id": "abc-123",
  "max_budget": 10.0,
  "budget_duration_sec": 2592000,
  "created_at": "2025-10-22T10:00:00Z",
  "updated_at": "2025-10-22T10:00:00Z"
}
```
</details>

## Assigning Budgets to Users

When creating or updating a user, specify the `budget_id`:

:::caution
If you don't create and set a budget, budget is unlimited.
:::

```bash
# Create a user with a budget
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-456",
    "alias": "Bob",
    "budget_id": "abc-123"
  }'

# Update an existing user's budget
curl -X PATCH http://localhost:8000/v1/users/user-123 \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"budget_id": "abc-123"}'
```

## Per-User Budget Resets

Budget resets are **per-user**, not global. Each user tracks their own budget period based on when they were assigned the budget.

**Example:**
1. Create a budget with `budget_duration_sec: 604800` (1 week)
2. Assign User A to the budget on Monday
3. Assign User B to the budget on Tuesday
4. User A's budget resets every Monday
5. User B's budget resets every Tuesday

This allows you to create budget tiers (like "Free", "Pro", "Enterprise") without worrying about all users resetting at the same time.

## Automatic Reset Behavior

Budget resets happen automatically using a "lazy reset" approach:
- When a user makes a request, the system checks if their `next_budget_reset_at` has passed
- If yes, the user's `spend` is reset to $0.00 and a new reset date is calculated
- A log entry is created in `budget_reset_logs` for audit purposes
- The request then proceeds normally

## Budget validation strategy

Budget enforcement is **enabled by default**. Every `/v1/chat/completions`, `/v1/messages`, and `/v1/embeddings` request checks the caller's budget limit and performs lazy resets on the way through.

How that check is serialized against concurrent budget resets is configurable via `budget_strategy`:

```bash
# env var
export GATEWAY_BUDGET_STRATEGY=cas

# or in config.yml
budget_strategy: cas
```

| Value | What it does | When to use |
|---|---|---|
| `for_update` (default) | `FOR UPDATE` acquired at the start and held across the entire request, including the LLM call. | Historical default — kept for backwards compatibility. Upgrade to `cas` when you can. |
| `cas` | Lock-free. Hot-path reads unlocked. Reset is an atomic conditional UPDATE (`WHERE next_budget_reset_at < now`), no explicit `FOR UPDATE`. | **Recommended.** Concurrent requests for the same user never serialize. |
| `disabled` | Skip `validate_user_budget` entirely — no user existence check, no blocked check, no budget check. | Usage-tracking-only deployments where budget/user enforcement happens out-of-band. Cost tracking via `log_usage` still runs. |

If you need user-blocked enforcement **without** budget checks, use `cas` and simply don't assign a `budget_id` to users — the gateway will still 404 on deleted users and 403 on blocked users while skipping all lock-taking budget logic.

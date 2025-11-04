# Budget Management

Budgets provide shared spending limits that can be assigned to multiple users. This allows you to create budget tiers (like "Free", "Pro", "Enterprise") and enforce spending limits across groups of users.

## Creating a Budget

```bash
# Create a budget with a $10.00 spending limit and monthly resets (30 days = 2592000 seconds)
curl -X POST http://localhost:8000/v1/budgets \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "max_budget": 10.0,
    "budget_duration_sec": 2592000
  }'
```

Response:
```json
{
  "budget_id": "abc-123",
  "max_budget": 10.0,
  "budget_duration_sec": 2592000,
  "created_at": "2025-10-22T10:00:00Z",
  "updated_at": "2025-10-22T10:00:00Z"
}
```

## Assigning Budgets to Users

When creating or updating a user, specify the `budget_id`:

**Warning: If you don't create and set a budget, budget is unlimited**

```bash
# Create a user with a budget
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-456",
    "alias": "Bob",
    "budget_id": "abc-123"
  }'

# Update an existing user's budget
curl -X PATCH http://localhost:8000/v1/users/user-123 \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"budget_id": "abc-123"}'
```

## Budget Enforcement

- **No budget assigned** (`budget_id: null`): User has unlimited spending
- **Budget with `max_budget: null`**: Budget tracks spending but doesn't enforce limits
- **Budget with `max_budget` set**: Requests are blocked when `user.spend >= max_budget`

## Managing Budgets

```bash
# List all budgets
curl http://localhost:8000/v1/budgets \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key"

# Get specific budget
curl http://localhost:8000/v1/budgets/abc-123 \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key"

# Update budget
curl -X PATCH http://localhost:8000/v1/budgets/abc-123 \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"max_budget": 25.0}'

# Delete budget
curl -X DELETE http://localhost:8000/v1/budgets/abc-123 \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key"
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

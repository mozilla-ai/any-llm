// k6 load test comparing sync vs async gateway throughput against a fake
// (noop) upstream LLM provider.
//
// Run: k6 run -e KEY=gw-xxx -e MASTER_KEY=... -e GATEWAY=http://localhost:4000 load_test.js
//
// Phases (scenarios run sequentially):
//   1. warmup:         low-VU warmup to let pools / JIT warm up
//   2. distinct_users: unique user_id per VU -> raw per-request overhead
//   3. same_user:      every VU hits one user_id -> triggers DB row-lock
//                      contention that serializes on sync gateways
//
// Lifecycle hooks:
//   setup():     pre-creates the users the scenarios will reference
//                (using the master key). Returns {users} to teardown.
//   teardown():  soft-deletes the users after the run.
//
// Env vars:
//   KEY            gateway API key (required, used by scenarios)
//   MASTER_KEY     gateway master key (required, used by setup/teardown)
//   GATEWAY        gateway base URL (default http://localhost:4000)
//   MODEL          model string (default "openai:fake")
//   VUS            virtual users per scenario (default 100)
//   DURATION       duration per main scenario (default 30s)

import http from 'k6/http';
import { check, fail } from 'k6';

const KEY = __ENV.KEY;
const MASTER_KEY = __ENV.MASTER_KEY;
const GATEWAY = __ENV.GATEWAY || 'http://localhost:4000';
const MODEL = __ENV.MODEL || 'openai:fake';
const VUS = parseInt(__ENV.VUS || '100', 10);
const DURATION = __ENV.DURATION || '30s';

if (!KEY) fail('KEY env var is required (gateway API key)');
if (!MASTER_KEY) fail('MASTER_KEY env var is required (gateway master key)');

const WARMUP_DURATION = '5s';
const WARMUP_VUS = Math.max(2, Math.floor(VUS / 10));

// Scenario timing:
//   warmup:          starts at t=0,   runs 5s
//   distinct_users:  starts at t=7s,  runs DURATION (default 30s)
//   same_user:       starts at t=42s, runs DURATION (default 30s)
const DISTINCT_START = '7s';
const SAME_START = '42s';

export const options = {
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(50)', 'p(95)', 'p(99)'],
  scenarios: {
    warmup: {
      executor: 'constant-vus',
      vus: WARMUP_VUS,
      duration: WARMUP_DURATION,
      exec: 'distinctUsers',
      gracefulStop: '1s',
    },
    distinct_users: {
      executor: 'constant-vus',
      vus: VUS,
      duration: DURATION,
      exec: 'distinctUsers',
      startTime: DISTINCT_START,
      gracefulStop: '5s',
    },
    same_user: {
      executor: 'constant-vus',
      vus: VUS,
      duration: DURATION,
      exec: 'sameUser',
      startTime: SAME_START,
      gracefulStop: '5s',
    },
  },
  thresholds: {
    // Keep main scenarios under 1% failure; warmup is not asserted.
    'http_req_failed{scenario:distinct_users}': ['rate<0.01'],
    'http_req_failed{scenario:same_user}': ['rate<0.01'],
    // Force per-scenario tagged submetrics so handleSummary can read them.
    'http_reqs{scenario:distinct_users}': ['rate>0'],
    'http_reqs{scenario:same_user}': ['rate>0'],
    'http_req_duration{scenario:distinct_users}': ['p(95)>=0'],
    'http_req_duration{scenario:same_user}': ['p(95)>=0'],
  },
};

const REQUEST_HEADERS = {
  'X-AnyLLM-Key': `Bearer ${KEY}`,
  'Content-Type': 'application/json',
};

const MASTER_HEADERS = {
  'X-AnyLLM-Key': `Bearer ${MASTER_KEY}`,
  'Content-Type': 'application/json',
};

function completionBody(userId) {
  return JSON.stringify({
    model: MODEL,
    messages: [{ role: 'user', content: 'x' }],
    user: userId,
  });
}

export function setup() {
  console.log(`[setup] creating ${VUS} distinct users + 1 shared user`);
  const users = [];
  for (let i = 1; i <= VUS; i++) {
    const uid = `loadtest-distinct-${i}`;
    const res = http.post(
      `${GATEWAY}/v1/users`,
      JSON.stringify({ user_id: uid, alias: `loadtest ${i}` }),
      { headers: MASTER_HEADERS }
    );
    if (res.status !== 200 && res.status !== 409) {
      fail(`setup: failed to create user ${uid}: status=${res.status} body=${res.body}`);
    }
    users.push(uid);
  }
  const sharedId = 'loadtest-shared';
  const res = http.post(
    `${GATEWAY}/v1/users`,
    JSON.stringify({ user_id: sharedId, alias: 'loadtest shared' }),
    { headers: MASTER_HEADERS }
  );
  if (res.status !== 200 && res.status !== 409) {
    fail(`setup: failed to create shared user: status=${res.status} body=${res.body}`);
  }
  users.push(sharedId);
  console.log(`[setup] created ${users.length} users`);
  return { users };
}

export function teardown(data) {
  console.log(`[teardown] soft-deleting ${data.users.length} users`);
  for (const uid of data.users) {
    http.del(`${GATEWAY}/v1/users/${uid}`, null, { headers: MASTER_HEADERS });
  }
}

export function distinctUsers() {
  const userId = `loadtest-distinct-${__VU}`;
  const res = http.post(`${GATEWAY}/v1/chat/completions`, completionBody(userId), {
    headers: REQUEST_HEADERS,
  });
  check(res, { '200': (r) => r.status === 200 });
}

export function sameUser() {
  const res = http.post(`${GATEWAY}/v1/chat/completions`, completionBody('loadtest-shared'), {
    headers: REQUEST_HEADERS,
  });
  check(res, { '200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  function fmt(v, digits) {
    return (v === undefined || v === null) ? '—' : v.toFixed(digits);
  }
  function row(label, scenario) {
    const reqs = data.metrics[`http_reqs{scenario:${scenario}}`];
    const dur = data.metrics[`http_req_duration{scenario:${scenario}}`];
    const failed = data.metrics[`http_req_failed{scenario:${scenario}}`];
    const rps = fmt(reqs?.values?.rate, 1);
    const p50 = fmt(dur?.values?.['p(50)'], 1);
    const p95 = fmt(dur?.values?.['p(95)'], 1);
    const p99 = fmt(dur?.values?.['p(99)'], 1);
    const failRate = failed?.values?.rate !== undefined && failed?.values?.rate !== null
      ? (failed.values.rate * 100).toFixed(2)
      : '—';
    return `  ${label.padEnd(18)} rps=${rps.padStart(7)}  p50=${p50.padStart(6)}ms  p95=${p95.padStart(6)}ms  p99=${p99.padStart(6)}ms  fail=${failRate}%`;
  }

  const lines = [
    '',
    '=== Gateway throughput summary ===',
    `  GATEWAY    ${GATEWAY}`,
    `  MODEL      ${MODEL}`,
    `  VUS        ${VUS}   DURATION ${DURATION}`,
    '',
    row('distinct_users', 'distinct_users'),
    row('same_user', 'same_user'),
    '',
  ];

  return {
    stdout: lines.join('\n'),
    'load_results.json': JSON.stringify(data, null, 2),
  };
}

# Server Dashboard

A live request-monitoring dashboard for `miles_agent_server.py`. It tracks every
incoming `/run` request — showing instance IDs, agent names, requester hostnames,
base URLs, response status (pending / completed / error), timing, and rewards —
grouped by server session so you can distinguish requests across server restarts.

## Architecture

```
miles_agent_server.py  (port 8080)
  ├── RequestLogger        INSERT/UPDATE →  /data/agent_server_dashboard/requests.db  (SQLite)
  └── spawns dashboard subprocess

Dashboard (FastAPI)    (port 8081, separate process)
        SELECT        ←  (same .db file)
        ├── GET /                        HTML single-page app (auto-refreshes)
        ├── GET /trace?trial_dir=...     Trace viewer page (per-request agent trace)
        ├── GET /api/entries?session_id= JSON list of request log entries
        ├── GET /api/sessions            JSON list of session summaries (GROUP BY)
        ├── GET /api/cpu                 JSON current CPU usage percentage
        ├── GET /api/disk                JSON disk usage (free GB, used %)
        └── GET /api/trace?trial_dir=... JSON trajectory data (normalized ATIF)
```

### Data flow

1. When a `/run` request arrives, the server **INSERTs** a row with
   `status = 'pending'` into the `requests` table.
2. When the request completes (or fails), the same row is **UPDATEd** to
   `status = 'completed'` or `status = 'error'` — no duplication.
   The `trial_dir` column stores the path to the trial output directory.
3. The dashboard queries the database with `WHERE session_id = ?` (indexed),
   so it only reads the rows for the selected session.

### Session isolation

Each server startup generates a `session_id` in `YYMMDD-hh-mm-ss` format
(Pacific Time). Every row includes this ID plus the `session_started_at`
timestamp. The dashboard provides a dropdown to switch between sessions.

### Trace viewing

Clicking a completed request row opens a trace viewer that renders the agent's
trajectory (tool calls, reasoning, outputs) in a readable, collapsible format.

Trajectory files are loaded from the `trial_dir` stored with each request.
The backend supports both native mini-swe-agent format and Harbor ATIF format
(with automatic normalization). Reward values are loaded from
`trial_dir/verifier/reward.txt` when available.

### Why SQLite (not JSONL)

- **Indexed queries** — `GET /api/entries?session_id=X` hits an index instead
  of scanning every row.
- **UPDATE in place** — no duplicate rows for pending → completed transitions.
- **Constant file size** — one row per request, no bloat.
- **Concurrent read+write** — WAL mode handles dashboard reads during writes.
- **Zero new dependencies** — `sqlite3` is in the Python stdlib.

## Quick start

### Automatic (recommended)

Running `miles_agent_server.py` automatically starts the dashboard:

```bash
# Dashboard mounted at /dashboard on the same port
python miles_agent_server.py --port 11000
# → Dashboard at http://0.0.0.0:11000/dashboard/

# Dashboard on a separate port
python miles_agent_server.py --port 11000 --dashboard-port 11100
# → Dashboard at http://0.0.0.0:11100/
```

### Standalone (development / read-only inspection)

```bash
python -m server_dashboard.dashboard \
  --log-path /data/agent_server_dashboard/requests.db \
  --port 11100
```

## Configuration

| Parameter / env-var        | CLI flag               | Default                                       |
|----------------------------|------------------------|-----------------------------------------------|
| Database path              | `--dashboard-log-path` | `/data/agent_server_dashboard/requests.db`    |
| Database path (env)        | `DASHBOARD_LOG_PATH`   | (same)                                        |
| Dashboard port (separate)  | `--dashboard-port`     | Not set (dashboard mounted on main server port)|

Priority: CLI flag > environment variable > built-in default.

The database path is **fixed across server sessions** by default — data
accumulates in the same file, and the dashboard uses `session_id` to filter
entries from different runs.

## Database schema

Single table `requests` with an index on `session_id`:

```sql
CREATE TABLE IF NOT EXISTS requests (
    request_id         TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    session_started_at TEXT NOT NULL,
    received_at        TEXT NOT NULL,
    instance_id        TEXT NOT NULL,
    requester_hostname TEXT NOT NULL,
    requester_port     TEXT NOT NULL,
    base_url           TEXT NOT NULL,
    model              TEXT DEFAULT '',
    agent_name         TEXT DEFAULT '',
    status             TEXT DEFAULT 'pending',
    responded_at       TEXT,
    duration_sec       REAL,
    reward             REAL,
    exit_status        TEXT,
    error_message      TEXT,
    trial_dir          TEXT
);
CREATE INDEX IF NOT EXISTS idx_session ON requests(session_id);
```

## Deployment

By default the dashboard runs in a **separate process** on port 8081
(`--dashboard-port 8081`).  This prevents agent workloads on the main
server from blocking dashboard responses.  Pass `--dashboard-port 0` to
fall back to mounting the dashboard at `/dashboard` on the main server.

## Dashboard UI features

- **Auto-refresh** — polls the API every N seconds (configurable: 2s / 5s / 10s / 30s).
- **Session dropdown** — select which server session to view; supports `?session=xxx` URL param.
- **Status filtering** — dropdown to show only pending / completed / error entries.
- **Search** — filter by instance ID, agent name, hostname, base URL, or request ID.
- **Summary cards** — CPU usage, disk free space, total, pending, completed, error counts for the selected session.
- **Clickable rows** — rows with a `trial_dir` are clickable and open the trace viewer.
- **Trace viewer** — renders agent trajectories with collapsible steps, thinking blocks,
  tool calls with labels, tool output with return code badges, diff highlighting,
  and a sticky step navigation sidebar.
- **Pacific Time** — all timestamps displayed in America/Los_Angeles.

## API reference

### `GET /api/entries?session_id=<id>`

Returns a JSON list of `RequestLogEntry` objects for the given session (latest 1000).

### `GET /api/sessions`

Returns a JSON list of `SessionInfo` objects with aggregate counts.

### `GET /api/cpu`

Returns `{"cpu_percent": <float|null>}`. Uses `/proc/stat` (Linux only).
Returns `null` on the first call (no delta yet) or on non-Linux systems.

### `GET /api/disk`

Returns `{"free_gb": <float>, "total_gb": <float>, "used_percent": <float>}` for
the root filesystem. Values are `null` if the filesystem cannot be read.

### `GET /api/trace?trial_dir=<path>`

Returns the normalized trajectory JSON for a given trial directory. Supports:
- Native mini-swe-agent format (`agent/mini-swe-agent.trajectory.json`)
- Harbor ATIF format (`agent/trajectory.json`) — automatically normalized

Response includes `instance_id`, `reward`, `info` (exit status, token stats),
and `messages` (system/user/assistant/tool messages).

## Module reference

### `server_dashboard.models`

| Class             | Description                                  |
|-------------------|----------------------------------------------|
| `RequestLogEntry` | Single request/response log record (Pydantic)|
| `SessionInfo`     | Aggregate session stats for `/api/sessions`  |

### `server_dashboard.request_logger`

| Class / method                                 | Description                                      |
|------------------------------------------------|--------------------------------------------------|
| `RequestLogger(log_path, session_id)`          | Thread-safe SQLite request logger                |
| `.log_request(...)`                            | INSERT a *pending* row                           |
| `.log_response(pending, ...)`                  | UPDATE the row to *completed*                    |
| `.log_error(pending, ...)`                     | UPDATE the row to *error*                        |
| `RequestLogger.read_entries(path)`             | SELECT all entries                               |
| `RequestLogger.read_entries_by_session(path, id)` | SELECT entries for a single session (indexed) |
| `RequestLogger.read_sessions(path)`            | Aggregate stats per session via GROUP BY         |

### `server_dashboard.dashboard`

| Function                             | Description                              |
|--------------------------------------|------------------------------------------|
| `create_dashboard_app(log_path)`     | Factory returning a mountable FastAPI app |
| `main()`                             | Standalone CLI entry point               |

## Running tests

```bash
# From the project root (pytest + server-dashboard are both available via uv)
uv run python -m pytest server-dashboard/tests/ -v
```

Tests use `tmp_path` fixtures with real SQLite databases — no running server or
Docker required.

## File structure

```
server-dashboard/
├── README.md                       # This file
├── pyproject.toml                  # Package metadata (uv path dependency)
├── server_dashboard/               # Python package
│   ├── __init__.py                 # Package exports
│   ├── models.py                   # Pydantic data models
│   ├── request_logger.py           # SQLite logging + query logic
│   ├── dashboard.py                # FastAPI app (API routes + static serving)
│   └── static/                     # Frontend assets
│       ├── index.html              # Dashboard page (request table)
│       ├── trace.html              # Trace viewer page (agent trajectory)
│       ├── style.css               # Dashboard styles
│       └── app.js                  # Client-side logic (fetch, render, filters)
└── tests/
    ├── __init__.py
    ├── test_models.py              # Model validation tests
    ├── test_request_logger.py      # SQLite logger tests (INSERT, UPDATE, queries)
    └── test_dashboard.py           # Dashboard API endpoint tests
```

"""SQLite database for storing request tracking."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from models import ScenarioID

DEFAULT_DB_PATH = Path(__file__).parent / "results.db"


class ResultsDB:
    """SQLite-based storage for request tracking."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    description TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id TEXT NOT NULL,
                    request_id TEXT,
                    scenario TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    request_body TEXT,
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_test_run
                ON requests(test_run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_scenario
                ON requests(scenario)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_test_run(
        self,
        test_run_id: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new test run."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO test_runs (id, created_at, description, metadata) VALUES (?, ?, ?, ?)",
                (
                    test_run_id,
                    time.time(),
                    description,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        return {
            "id": test_run_id,
            "created_at": time.time(),
            "description": description,
        }

    def get_test_run(self, test_run_id: str) -> dict[str, Any] | None:
        """Get a test run by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM test_runs WHERE id = ?", (test_run_id,)
            ).fetchone()
            if row:
                return {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "description": row["description"],
                    "metadata": json.loads(row["metadata"])
                    if row["metadata"]
                    else None,
                }
        return None

    def list_test_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all test runs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM test_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "description": row["description"],
                    "metadata": json.loads(row["metadata"])
                    if row["metadata"]
                    else None,
                }
                for row in rows
            ]

    def delete_test_run(self, test_run_id: str) -> bool:
        """Delete a test run and its requests."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM requests WHERE test_run_id = ?", (test_run_id,))
            cursor = conn.execute("DELETE FROM test_runs WHERE id = ?", (test_run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def store_request(
        self,
        test_run_id: str,
        scenario: ScenarioID,
        request_id: str,
        request_body: dict[str, Any] | None = None,
    ) -> None:
        """Store a request."""
        with self._get_connection() as conn:
            run_exists = conn.execute(
                "SELECT 1 FROM test_runs WHERE id = ?", (test_run_id,)
            ).fetchone()
            if not run_exists:
                conn.execute(
                    "INSERT INTO test_runs (id, created_at, description, metadata) VALUES (?, ?, ?, ?)",
                    (test_run_id, time.time(), None, None),
                )

            conn.execute(
                """
                INSERT INTO requests
                (test_run_id, request_id, scenario, timestamp, request_body)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    test_run_id,
                    request_id,
                    scenario.value,
                    time.time(),
                    json.dumps(request_body) if request_body else None,
                ),
            )
            conn.commit()

    def get_requests(
        self,
        test_run_id: str | None = None,
        scenario: ScenarioID | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query requests with optional filters."""
        query = "SELECT * FROM requests WHERE 1=1"
        params: list[Any] = []

        if test_run_id is not None:
            query += " AND test_run_id = ?"
            params.append(test_run_id)

        if scenario is not None:
            query += " AND scenario = ?"
            params.append(scenario.value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                results.append(
                    {
                        "id": row["id"],
                        "test_run_id": row["test_run_id"],
                        "request_id": row["request_id"],
                        "scenario": row["scenario"],
                        "timestamp": row["timestamp"],
                        "request_body": json.loads(row["request_body"])
                        if row["request_body"]
                        else None,
                    }
                )
            return results

    def get_summary(self, test_run_id: str | None = None) -> dict[str, Any]:
        """Get a summary of requests, optionally filtered by test run."""
        query_base = "SELECT scenario, COUNT(*) as count FROM requests"
        params: list[Any] = []

        if test_run_id:
            query_base += " WHERE test_run_id = ?"
            params.append(test_run_id)

        query_base += " GROUP BY scenario"

        with self._get_connection() as conn:
            rows = conn.execute(query_base, params).fetchall()

            summary: dict[str, int] = {}
            total = 0

            for row in rows:
                scenario = row["scenario"]
                count = row["count"]
                summary[scenario] = count
                total += count

            return {
                "total": total,
                "by_scenario": summary,
            }

    def clear_all(self) -> None:
        """Clear all data from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM requests")
            conn.execute("DELETE FROM test_runs")
            conn.commit()


db = ResultsDB()

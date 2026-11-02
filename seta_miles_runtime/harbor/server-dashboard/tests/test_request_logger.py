"""Tests for server_dashboard.request_logger (SQLite backend)."""

import re
import sqlite3
from pathlib import Path

import pytest
from server_dashboard.request_logger import RequestLogger


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_requests.db"


@pytest.fixture()
def logger_instance(db_path: Path) -> RequestLogger:
    return RequestLogger(log_path=db_path, session_id="test-session-aaa")


class TestRequestLoggerInit:
    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "log.db"
        rl = RequestLogger(log_path=deep)
        assert deep.parent.exists()
        assert re.match(r"^\d{6}-\d{2}-\d{2}-\d{2}$", rl.session_id)

    def test_custom_session_id(self, db_path: Path):
        rl = RequestLogger(log_path=db_path, session_id="custom-123")
        assert rl.session_id == "custom-123"

    def test_creates_schema(self, db_path: Path):
        RequestLogger(log_path=db_path)
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert ("requests",) in tables
        conn.close()

    def test_wal_mode(self, db_path: Path):
        RequestLogger(log_path=db_path)
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestLogRequest:
    def test_inserts_pending_row(self, logger_instance: RequestLogger, db_path: Path):
        entry = logger_instance.log_request(
            request_id="req-001",
            instance_id="django__django-12345",
            requester_hostname="10.0.0.5",
            requester_port="9999",
            base_url="http://llm:8000/v1",
            model="openai/llama",
            agent_name="mini-swe-agent",
        )
        assert entry.status == "pending"
        assert entry.session_id == "test-session-aaa"
        assert entry.instance_id == "django__django-12345"

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status, request_id FROM requests WHERE request_id='req-001'"
        ).fetchone()
        assert row == ("pending", "req-001")
        conn.close()

    def test_multiple_requests(self, logger_instance: RequestLogger, db_path: Path):
        for i in range(5):
            logger_instance.log_request(
                request_id=f"req-{i:03d}",
                instance_id=f"task-{i}",
                requester_hostname="host",
                requester_port="80",
                base_url="http://x",
            )
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 5
        conn.close()


class TestLogResponse:
    def test_updates_to_completed(self, logger_instance: RequestLogger, db_path: Path):
        pending = logger_instance.log_request(
            request_id="req-010",
            instance_id="task-A",
            requester_hostname="host",
            requester_port="80",
            base_url="http://x",
        )
        completed = logger_instance.log_response(
            pending, reward=1.0, exit_status="Submitted"
        )
        assert completed.status == "completed"
        assert completed.reward == 1.0
        assert completed.exit_status == "Submitted"
        assert completed.duration_sec is not None
        assert completed.responded_at is not None

        # Only 1 row — UPDATE, not a second INSERT
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 1
        row = conn.execute(
            "SELECT status, reward FROM requests WHERE request_id='req-010'"
        ).fetchone()
        assert row == ("completed", 1.0)
        conn.close()


class TestLogError:
    def test_updates_to_error(self, logger_instance: RequestLogger, db_path: Path):
        pending = logger_instance.log_request(
            request_id="req-020",
            instance_id="task-B",
            requester_hostname="host",
            requester_port="80",
            base_url="http://x",
        )
        errored = logger_instance.log_error(pending, error_message="Docker crash")
        assert errored.status == "error"
        assert errored.error_message == "Docker crash"
        assert errored.duration_sec is not None

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 1
        conn.close()


class TestReadEntries:
    def test_nonexistent_db(self, tmp_path: Path):
        p = tmp_path / "nope.db"
        assert RequestLogger.read_entries(p) == []

    def test_returns_latest_status(self, logger_instance: RequestLogger, db_path: Path):
        pending = logger_instance.log_request(
            request_id="req-100",
            instance_id="task-X",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        logger_instance.log_response(pending, reward=0.5, exit_status="Submitted")

        entries = RequestLogger.read_entries(db_path)
        assert len(entries) == 1
        assert entries[0].status == "completed"
        assert entries[0].reward == 0.5

    def test_multiple_requests(self, logger_instance: RequestLogger, db_path: Path):
        for i in range(3):
            p = logger_instance.log_request(
                request_id=f"req-{i}",
                instance_id=f"task-{i}",
                requester_hostname="h",
                requester_port="80",
                base_url="http://x",
            )
            logger_instance.log_response(p, reward=float(i), exit_status="Done")
        entries = RequestLogger.read_entries(db_path)
        assert len(entries) == 3

    def test_multi_session_entries(self, db_path: Path):
        rl1 = RequestLogger(log_path=db_path, session_id="session-AAA")
        rl2 = RequestLogger(log_path=db_path, session_id="session-BBB")

        p1 = rl1.log_request(
            request_id="req-a",
            instance_id="task-1",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        rl1.log_response(p1, reward=1.0, exit_status="OK")

        p2 = rl2.log_request(
            request_id="req-b",
            instance_id="task-2",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        rl2.log_error(p2, error_message="fail")

        entries = RequestLogger.read_entries(db_path)
        assert len(entries) == 2
        sessions = {e.session_id for e in entries}
        assert sessions == {"session-AAA", "session-BBB"}


class TestReadEntriesBySession:
    def test_returns_only_matching_session(self, db_path: Path):
        rl1 = RequestLogger(log_path=db_path, session_id="sess-A")
        rl2 = RequestLogger(log_path=db_path, session_id="sess-B")

        for i in range(3):
            rl1.log_request(
                request_id=f"a-{i}",
                instance_id=f"task-a-{i}",
                requester_hostname="h",
                requester_port="80",
                base_url="http://x",
            )
        for i in range(2):
            rl2.log_request(
                request_id=f"b-{i}",
                instance_id=f"task-b-{i}",
                requester_hostname="h",
                requester_port="80",
                base_url="http://x",
            )

        entries_a = RequestLogger.read_entries_by_session(db_path, "sess-A")
        assert len(entries_a) == 3
        assert all(e.session_id == "sess-A" for e in entries_a)

        entries_b = RequestLogger.read_entries_by_session(db_path, "sess-B")
        assert len(entries_b) == 2

    def test_nonexistent_session(self, logger_instance: RequestLogger, db_path: Path):
        logger_instance.log_request(
            request_id="r1",
            instance_id="t1",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        entries = RequestLogger.read_entries_by_session(db_path, "no-such-session")
        assert entries == []

    def test_nonexistent_db(self, tmp_path: Path):
        entries = RequestLogger.read_entries_by_session(tmp_path / "nope.db", "x")
        assert entries == []


class TestReadSessions:
    def test_aggregate_counts(self, db_path: Path):
        rl = RequestLogger(log_path=db_path, session_id="sess-X")
        p1 = rl.log_request(
            request_id="r1",
            instance_id="t1",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        rl.log_response(p1, reward=1.0, exit_status="OK")

        p2 = rl.log_request(
            request_id="r2",
            instance_id="t2",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        rl.log_error(p2, error_message="boom")

        rl.log_request(
            request_id="r3",
            instance_id="t3",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )

        sessions = RequestLogger.read_sessions(db_path)
        assert len(sessions) == 1
        s = sessions[0]
        assert s.session_id == "sess-X"
        assert s.request_count == 3
        assert s.completed_count == 1
        assert s.error_count == 1
        assert s.pending_count == 1

    def test_multiple_sessions(self, db_path: Path):
        rl1 = RequestLogger(log_path=db_path, session_id="s1")
        rl2 = RequestLogger(log_path=db_path, session_id="s2")
        rl1.log_request(
            request_id="a",
            instance_id="t",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        rl2.log_request(
            request_id="b",
            instance_id="t",
            requester_hostname="h",
            requester_port="80",
            base_url="http://x",
        )
        sessions = RequestLogger.read_sessions(db_path)
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}

    def test_nonexistent_db(self, tmp_path: Path):
        sessions = RequestLogger.read_sessions(tmp_path / "nope.db")
        assert sessions == []

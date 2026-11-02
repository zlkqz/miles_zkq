"""Tests for server_dashboard.models."""

import pytest
from server_dashboard.models import RequestLogEntry, SessionInfo


class TestRequestLogEntry:
    """Validate RequestLogEntry construction and defaults."""

    def _make_entry(self, **overrides) -> RequestLogEntry:
        defaults = {
            "session_id": "sess-001",
            "session_started_at": "2025-07-01T00:00:00+00:00",
            "request_id": "req-001",
            "received_at": "2025-07-01T00:01:00+00:00",
            "instance_id": "django__django-12345",
            "requester_hostname": "10.0.0.1",
            "requester_port": "54321",
            "base_url": "http://localhost:8000/v1",
        }
        defaults.update(overrides)
        return RequestLogEntry(**defaults)

    def test_default_status_is_pending(self):
        entry = self._make_entry()
        assert entry.status == "pending"

    def test_optional_fields_default_to_none(self):
        entry = self._make_entry()
        assert entry.responded_at is None
        assert entry.duration_sec is None
        assert entry.reward is None
        assert entry.exit_status is None
        assert entry.error_message is None

    def test_model_and_agent_defaults(self):
        entry = self._make_entry()
        assert entry.model == ""
        assert entry.agent_name == ""

    def test_all_fields_populated(self):
        entry = self._make_entry(
            status="completed",
            responded_at="2025-07-01T00:05:00+00:00",
            duration_sec=240.0,
            reward=1.0,
            exit_status="Submitted",
            model="openai/llama",
            agent_name="mini-swe-agent",
        )
        assert entry.status == "completed"
        assert entry.reward == 1.0
        assert entry.duration_sec == 240.0

    def test_json_round_trip(self):
        entry = self._make_entry(status="error", error_message="boom")
        raw = entry.model_dump_json()
        restored = RequestLogEntry.model_validate_json(raw)
        assert restored == entry

    def test_invalid_missing_required(self):
        with pytest.raises(Exception):
            RequestLogEntry(session_id="x")  # missing many required fields


class TestSessionInfo:
    """Validate SessionInfo construction."""

    def test_defaults(self):
        si = SessionInfo(session_id="s1", session_started_at="2025-01-01T00:00:00Z")
        assert si.request_count == 0
        assert si.completed_count == 0
        assert si.error_count == 0
        assert si.pending_count == 0

    def test_mutable_counts(self):
        si = SessionInfo(session_id="s1", session_started_at="2025-01-01T00:00:00Z")
        si.request_count = 10
        si.completed_count = 7
        si.error_count = 2
        si.pending_count = 1
        data = si.model_dump()
        assert data["request_count"] == 10
        assert data["completed_count"] == 7

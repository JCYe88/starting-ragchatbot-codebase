"""
API endpoint tests for the FastAPI routes defined in app.py.

Uses an inline test app (via conftest fixtures) to avoid the module-level
StaticFiles mount in app.py that requires ../frontend to exist at import time.

Covers:
  POST /api/query   — query processing, session handling, error propagation
  GET  /api/courses — course statistics
  DELETE /api/session/{id} — session cleanup
"""

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/query
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryEndpoint:

    def test_success_returns_200(self, client):
        resp = client.post("/api/query", json={"query": "What is deep learning?"})
        assert resp.status_code == 200

    def test_response_contains_answer(self, client):
        resp = client.post("/api/query", json={"query": "What is deep learning?"})
        assert resp.json()["answer"] == "Test answer"

    def test_response_contains_session_id(self, client):
        resp = client.post("/api/query", json={"query": "question"})
        assert resp.json()["session_id"] == "test-session-id"

    def test_response_contains_sources(self, client):
        resp = client.post("/api/query", json={"query": "question"})
        sources = resp.json()["sources"]
        assert isinstance(sources, list)
        assert len(sources) == 1
        assert sources[0]["label"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"

    def test_auto_creates_session_when_none_provided(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "question"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id_without_creating_new(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "question", "session_id": "my-session"})
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_provided_session_id_forwarded_to_rag(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "question", "session_id": "my-session"})
        positional_args = mock_rag_system.query.call_args[0]
        assert positional_args[1] == "my-session"

    def test_rag_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        resp = client.post("/api/query", json={"query": "anything"})
        assert resp.status_code == 500

    def test_rag_exception_detail_in_response(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        resp = client.post("/api/query", json={"query": "anything"})
        assert "DB unavailable" in resp.json()["detail"]

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "x"})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_empty_string_query_accepted(self, client):
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/courses
# ══════════════════════════════════════════════════════════════════════════════

class TestCoursesEndpoint:

    def test_success_returns_200(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_returns_total_courses_count(self, client):
        resp = client.get("/api/courses")
        assert resp.json()["total_courses"] == 2

    def test_returns_course_titles_list(self, client):
        resp = client.get("/api/courses")
        titles = resp.json()["course_titles"]
        assert "Course A" in titles
        assert "Course B" in titles

    def test_rag_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("analytics error")
        resp = client.get("/api/courses")
        assert resp.status_code == 500

    def test_rag_exception_detail_in_response(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("analytics error")
        resp = client.get("/api/courses")
        assert "analytics error" in resp.json()["detail"]

    def test_empty_course_list_is_valid(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        assert resp.json()["total_courses"] == 0
        assert resp.json()["course_titles"] == []


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /api/session/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteSessionEndpoint:

    def test_delete_returns_200(self, client):
        resp = client.delete("/api/session/abc-123")
        assert resp.status_code == 200

    def test_delete_returns_ok_status(self, client):
        resp = client.delete("/api/session/abc-123")
        assert resp.json() == {"status": "ok"}

    def test_delete_calls_session_manager_with_correct_id(self, client, mock_rag_system):
        client.delete("/api/session/my-session-id")
        mock_rag_system.session_manager.delete_session.assert_called_once_with("my-session-id")

    def test_delete_nonexistent_session_still_returns_200(self, client):
        resp = client.delete("/api/session/does-not-exist")
        assert resp.status_code == 200

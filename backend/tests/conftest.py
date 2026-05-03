"""
Shared fixtures and path setup for all tests.
Tests are run from backend/ so imports resolve the same way the app does.
"""
import sys
import os

# Ensure the backend package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


@pytest.fixture
def mock_rag_system():
    """RAGSystem mock with sensible defaults for API-level tests."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = (
        "Test answer",
        [{"label": "Course A - Lesson 1", "url": "https://example.com/lesson1"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Minimal FastAPI app mirroring the real API routes in app.py.

    Defined here (not imported from app.py) to avoid the module-level
    StaticFiles mount that requires ../frontend to exist at import time.
    """

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        label: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = (
                request.session_id
                or mock_rag_system.session_manager.create_session()
            )
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag_system.session_manager.delete_session(session_id)
        return {"status": "ok"}

    return app


@pytest.fixture
def client(test_app):
    """TestClient wrapping the test_app fixture."""
    return TestClient(test_app)

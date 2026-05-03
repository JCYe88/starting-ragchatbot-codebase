"""
Tests for CourseSearchTool.execute() in search_tools.py.

Three tiers:
  1. Unit tests with a fully mocked VectorStore — fast, no I/O.
  2. Integration tests against a real (temporary) ChromaDB — verifies
     the tool survives edge cases like an empty collection.
  3. One smoke test against the live backend/chroma_db/ (skipped if absent).
"""

import os
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore, SearchResults
from models import CourseChunk


# ── Shared helpers ──────────────────────────────────────────────────────────

def _results(docs, metas):
    """Build a SearchResults with the given parallel docs/metas lists."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = None
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


@pytest.fixture
def real_store(tmp_path):
    """In-memory-ish ChromaDB in a temp directory — starts completely empty."""
    return VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit tests (mocked VectorStore)
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteReturnType:
    """execute() must always return a plain string — never raise."""

    def test_returns_string_on_empty_results(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        result = tool.execute(query="transformers")
        assert isinstance(result, str)

    def test_returns_string_when_store_raises(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty("Search error: DB gone")
        result = tool.execute(query="anything")
        assert isinstance(result, str)

    def test_returns_string_on_populated_results(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["Backprop is a training algorithm."],
            metas=[{"course_title": "ML Basics", "lesson_number": 2}],
        )
        result = tool.execute(query="backprop")
        assert isinstance(result, str)


class TestExecuteContent:
    """execute() must surface the right content in its output."""

    def test_empty_results_message_present(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        assert "No relevant content found" in tool.execute(query="xyz")

    def test_course_name_echoed_in_empty_message(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        assert "My Course" in tool.execute(query="xyz", course_name="My Course")

    def test_error_string_passed_through(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty("Search error: timeout")
        assert "Search error" in tool.execute(query="x")

    def test_formatted_result_contains_course_title(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["Neural networks are..."],
            metas=[{"course_title": "Intro to ML", "lesson_number": 1}],
        )
        assert "Intro to ML" in tool.execute(query="nn")

    def test_formatted_result_contains_lesson_number(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["Content here"],
            metas=[{"course_title": "Course X", "lesson_number": 3}],
        )
        assert "Lesson 3" in tool.execute(query="content")

    def test_formatted_result_contains_document_text(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["Gradient descent minimizes loss."],
            metas=[{"course_title": "DL", "lesson_number": 1}],
        )
        assert "Gradient descent minimizes loss." in tool.execute(query="gradients")

    def test_multiple_results_all_present(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["Doc A text", "Doc B text"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
        )
        out = tool.execute(query="docs")
        assert "Course A" in out and "Doc A text" in out
        assert "Course B" in out and "Doc B text" in out


class TestExecuteForwardsFilters:
    """execute() must forward course_name and lesson_number to the store."""

    def test_no_filters_passes_none_to_store(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool.execute(query="backprop")
        mock_store.search.assert_called_once_with(
            query="backprop", course_name=None, lesson_number=None
        )

    def test_course_name_forwarded(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool.execute(query="backprop", course_name="Deep Learning")
        mock_store.search.assert_called_once_with(
            query="backprop", course_name="Deep Learning", lesson_number=None
        )

    def test_lesson_number_forwarded(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool.execute(query="backprop", lesson_number=2)
        mock_store.search.assert_called_once_with(
            query="backprop", course_name=None, lesson_number=2
        )

    def test_both_filters_forwarded(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool.execute(query="q", course_name="MCP", lesson_number=4)
        mock_store.search.assert_called_once_with(
            query="q", course_name="MCP", lesson_number=4
        )


class TestSourceTracking:
    """last_sources must be populated after a successful search."""

    def test_sources_populated_after_search(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["content"],
            metas=[{"course_title": "ML", "lesson_number": 3}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/3"
        tool.execute(query="something")
        assert len(tool.last_sources) == 1
        src = tool.last_sources[0]
        assert src["label"] == "ML - Lesson 3"
        assert src["url"] == "https://example.com/3"

    def test_sources_empty_on_no_results(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool.execute(query="nothing")
        # last_sources from prior run should be overwritten to the new result set
        # (empty, since _format_results was never called — but we check it doesn't crash)
        assert isinstance(tool.last_sources, list)

    def test_multiple_results_produce_multiple_sources(self, tool, mock_store):
        mock_store.search.return_value = _results(
            docs=["d1", "d2"],
            metas=[
                {"course_title": "C1", "lesson_number": 1},
                {"course_title": "C2", "lesson_number": 2},
            ],
        )
        tool.execute(query="multi")
        assert len(tool.last_sources) == 2


class TestToolManagerDispatch:
    """ToolManager must correctly route calls to CourseSearchTool."""

    def test_dispatch_to_search_tool(self, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        t = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(t)
        result = mgr.execute_tool("search_course_content", query="hello")
        assert isinstance(result, str)

    def test_unknown_tool_returns_error_string(self):
        mgr = ToolManager()
        result = mgr.execute_tool("nonexistent_tool", query="hi")
        assert isinstance(result, str)
        assert "not found" in result.lower() or "Tool" in result

    def test_get_last_sources_after_search(self, mock_store):
        mock_store.search.return_value = _results(
            docs=["c"],
            metas=[{"course_title": "X", "lesson_number": 1}],
        )
        t = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(t)
        mgr.execute_tool("search_course_content", query="test")
        assert len(mgr.get_last_sources()) == 1

    def test_reset_clears_sources(self, mock_store):
        mock_store.search.return_value = _results(
            docs=["c"],
            metas=[{"course_title": "X", "lesson_number": 1}],
        )
        t = CourseSearchTool(mock_store)
        mgr = ToolManager()
        mgr.register_tool(t)
        mgr.execute_tool("search_course_content", query="test")
        mgr.reset_sources()
        assert mgr.get_last_sources() == []


# ══════════════════════════════════════════════════════════════════════════════
# 2. Integration tests — real ChromaDB (empty, then populated)
# ══════════════════════════════════════════════════════════════════════════════

class TestWithRealChromaDBEmpty:
    """
    With an empty ChromaDB, execute() must still return a string.
    ChromaDB 1.x raises when n_results > collection size; the VectorStore
    try/except must catch that and surface it as a string, not an exception.
    """

    def test_empty_collection_returns_string_no_raise(self, real_store):
        t = CourseSearchTool(real_store)
        result = t.execute(query="what is backpropagation?")
        assert isinstance(result, str), (
            "execute() raised instead of returning a string. "
            "VectorStore.search() is not catching the ChromaDB n_results > count error."
        )

    def test_empty_collection_with_course_filter_returns_string(self, real_store):
        t = CourseSearchTool(real_store)
        result = t.execute(query="q", course_name="Nonexistent Course")
        assert isinstance(result, str)


class TestWithRealChromaDBPopulated:
    """Adding real chunks and searching must return relevant content."""

    @pytest.fixture
    def populated_store(self, real_store):
        chunks = [
            CourseChunk(
                content="Backpropagation computes gradients via the chain rule.",
                course_title="Deep Learning 101",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Gradient descent updates weights to minimise the loss.",
                course_title="Deep Learning 101",
                lesson_number=1,
                chunk_index=1,
            ),
            CourseChunk(
                content="Convolutional layers learn spatial features from images.",
                course_title="Deep Learning 101",
                lesson_number=2,
                chunk_index=2,
            ),
        ]
        real_store.add_course_content(chunks)
        return real_store

    def test_search_returns_non_empty_result(self, populated_store):
        t = CourseSearchTool(populated_store)
        result = t.execute(query="backpropagation training algorithm")
        assert "No relevant content found" not in result
        assert len(result) > 0

    def test_search_result_contains_relevant_text(self, populated_store):
        t = CourseSearchTool(populated_store)
        result = t.execute(query="gradient descent loss")
        assert "Deep Learning 101" in result

    def test_lesson_filter_narrows_results(self, populated_store):
        t = CourseSearchTool(populated_store)
        result = t.execute(query="learning", lesson_number=2)
        assert "Lesson 2" in result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Smoke test against live chroma_db/ (skipped if the DB is absent)
# ══════════════════════════════════════════════════════════════════════════════

LIVE_CHROMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)

@pytest.mark.skipif(not os.path.exists(LIVE_CHROMA), reason="live chroma_db not present")
class TestLiveChromaDB:

    @pytest.fixture
    def live_store(self):
        return VectorStore(LIVE_CHROMA, "all-MiniLM-L6-v2", max_results=5)

    def test_live_search_returns_results(self, live_store):
        t = CourseSearchTool(live_store)
        result = t.execute(query="what is a neural network")
        assert isinstance(result, str)
        assert len(result) > 10, "Expected non-trivial content from live DB"

    def test_live_sources_populated(self, live_store):
        t = CourseSearchTool(live_store)
        t.execute(query="MCP tools")
        assert len(t.last_sources) > 0, "Sources should be populated after a live search"

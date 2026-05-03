"""
Tests for RAGSystem.query() focused on content-related question handling.

Failure taxonomy this suite catches:
  A. RAGSystem doesn't pass the tool_manager to generate_response
     → Claude can use tools but execute_tool never runs
     → sources always empty; tool result never reaches Claude
  B. generate_response raises an unhandled exception
     → app.py catches it, returns HTTP 500, frontend shows "Query failed"
  C. VectorStore search failure escapes the try/except
     → same HTTP 500 path
  D. Session handling corrupts the conversation or raises on re-use
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from models import CourseChunk


# ── Helpers ──────────────────────────────────────────────────────────────────

def _text_block(text):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_block(name, tool_id, input_dict):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = input_dict
    return b


def _direct_response(text):
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [_text_block(text)]
    return r


def _tool_use_response(name, tool_id, input_dict):
    r = MagicMock()
    r.stop_reason = "tool_use"
    r.content = [_tool_block(name, tool_id, input_dict)]
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rag_mocked(tmp_path):
    """
    RAGSystem with:
      - Real SessionManager (exercises history logic)
      - Real ToolManager + a mock CourseSearchTool
      - Real AIGenerator object but with its Anthropic client swapped for a mock
      - Real VectorStore object but with its ChromaDB pointed at an empty tmp dir
    """
    from config import Config
    from ai_generator import AIGenerator
    from session_manager import SessionManager
    from vector_store import VectorStore
    from document_processor import DocumentProcessor

    cfg = Config(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="test-model",
        CHROMA_PATH=str(tmp_path / "chroma"),
    )

    # Build components directly (avoids patching at import level)
    rag = object.__new__(RAGSystem)
    rag.config = cfg
    rag.document_processor = DocumentProcessor(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    rag.vector_store = VectorStore(cfg.CHROMA_PATH, cfg.EMBEDDING_MODEL, cfg.MAX_RESULTS)
    rag.session_manager = SessionManager(cfg.MAX_HISTORY)

    # Real AIGenerator but with a mock client
    gen = object.__new__(AIGenerator)
    gen.model = cfg.ANTHROPIC_MODEL
    gen.base_params = {"model": cfg.ANTHROPIC_MODEL, "temperature": 0, "max_tokens": 800}
    gen.client = MagicMock()
    rag.ai_generator = gen

    # ToolManager with a mock search tool
    mock_search = MagicMock(spec=CourseSearchTool)
    mock_search.get_tool_definition.return_value = {
        "name": "search_course_content",
        "description": "Search course content",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    mock_search.execute.return_value = "Lesson content: neural networks explained."
    mock_search.last_sources = [{"label": "Course A - Lesson 1", "url": "https://example.com"}]
    rag.search_tool = mock_search
    rag.tool_manager = ToolManager()
    rag.tool_manager.tools["search_course_content"] = mock_search

    return rag


# ══════════════════════════════════════════════════════════════════════════════
# A. Return-shape and routing
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryReturnShape:

    def test_returns_two_tuple(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        result = rag_mocked.query("What is a neural network?")
        assert isinstance(result, tuple) and len(result) == 2

    def test_answer_is_string(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("An answer.")
        answer, _ = rag_mocked.query("question")
        assert isinstance(answer, str) and len(answer) > 0

    def test_sources_is_list(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        _, sources = rag_mocked.query("question")
        assert isinstance(sources, list)


# ══════════════════════════════════════════════════════════════════════════════
# B. Tool manager wiring — the most failure-prone integration point
# ══════════════════════════════════════════════════════════════════════════════

class TestToolManagerWiring:

    def test_tool_definitions_passed_to_generate_response(self, rag_mocked):
        """
        AIGenerator.generate_response must receive the tools list.
        If it doesn't, Claude never knows the tool exists and never calls it.
        """
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        rag_mocked.query("What is taught in lesson 2?")

        # generate_response is the real method, so we check via the mock client
        first_call_kwargs = rag_mocked.ai_generator.client.messages.create.call_args[1]
        assert "tools" in first_call_kwargs, (
            "The tools list was not passed to the Anthropic API call. "
            "Claude has no knowledge of the search tool and cannot use it."
        )
        tool_names = [t["name"] for t in first_call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_tool_manager_passed_so_tool_can_be_executed(self, rag_mocked):
        """
        When Claude returns tool_use, _handle_tool_execution must have a
        tool_manager to call. If tool_manager=None is passed, the tool is
        never executed and the final answer has no retrieved content.
        """
        rag_mocked.ai_generator.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_01", {"query": "neural net"}),
            _direct_response("Neural networks are..."),
        ]
        answer, _ = rag_mocked.query("What is a neural network in the course?")
        # If tool_manager was None, execute would never have been called
        rag_mocked.search_tool.execute.assert_called_once()

    def test_content_query_triggers_search_tool_execution(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_02", {"query": "backprop algorithm"}),
            _direct_response("Backpropagation works by..."),
        ]
        answer, _ = rag_mocked.query("How does backpropagation work in the course?")
        assert answer == "Backpropagation works by..."
        rag_mocked.search_tool.execute.assert_called_once_with(query="backprop algorithm")

    def test_sources_returned_after_tool_use(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_03", {"query": "x"}),
            _direct_response("answer"),
        ]
        _, sources = rag_mocked.query("course-specific question")
        assert len(sources) > 0, (
            "Sources should be non-empty after a search tool was used. "
            "Check that ToolManager.get_last_sources() is called and that "
            "reset_sources() does not run before sources are collected."
        )

    def test_sources_reset_between_queries(self, rag_mocked):
        """Sources from query N must not bleed into query N+1."""
        rag_mocked.ai_generator.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_04", {"query": "x"}),
            _direct_response("first answer"),
            _direct_response("second answer"),  # no tool use this time
        ]
        rag_mocked.query("first")
        # Reset the mock's last_sources to simulate no tool call on second query
        rag_mocked.search_tool.last_sources = []
        _, sources2 = rag_mocked.query("second")
        assert sources2 == [], "Sources from the first query bled into the second"


# ══════════════════════════════════════════════════════════════════════════════
# C. Exception propagation — reproduces the "Query failed" frontend message
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptionPropagation:

    def test_api_exception_propagates_from_query(self, rag_mocked):
        """
        If the Anthropic API raises (e.g. model deprecated, auth failure),
        the exception must propagate out of rag.query() so app.py can catch
        it and return HTTP 500.  This is the exact failure that produces
        'Query failed' in the frontend.
        """
        rag_mocked.ai_generator.client.messages.create.side_effect = Exception(
            "model_not_found: claude-sonnet-4-20250514 is deprecated"
        )
        with pytest.raises(Exception, match="model_not_found"):
            rag_mocked.query("Anything about the course")

    def test_empty_content_raises_descriptive_value_error(self, rag_mocked):
        """
        If the Anthropic API returns an empty content list, the guard in
        generate_response() must raise a ValueError with a useful message
        rather than a bare IndexError, so the operator can diagnose it.
        """
        empty_resp = MagicMock()
        empty_resp.stop_reason = "end_turn"
        empty_resp.content = []

        rag_mocked.ai_generator.client.messages.create.return_value = empty_resp
        with pytest.raises(ValueError, match="empty content"):
            rag_mocked.query("What does lesson 3 cover?")


# ══════════════════════════════════════════════════════════════════════════════
# D. Session handling
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionHandling:

    def test_query_without_session_id_does_not_raise(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        answer, _ = rag_mocked.query("general question")
        assert answer == "ok"

    def test_query_with_session_id_saves_history(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        session_id = rag_mocked.session_manager.create_session()
        rag_mocked.query("first question", session_id=session_id)
        history = rag_mocked.session_manager.get_conversation_history(session_id)
        assert history is not None and len(history) > 0

    def test_history_passed_on_followup(self, rag_mocked):
        rag_mocked.ai_generator.client.messages.create.return_value = _direct_response("ok")
        session_id = rag_mocked.session_manager.create_session()
        rag_mocked.query("first question", session_id=session_id)
        rag_mocked.query("follow up", session_id=session_id)
        # Second call should include system prompt with history
        second_call_kwargs = rag_mocked.ai_generator.client.messages.create.call_args[1]
        assert "first question" in second_call_kwargs["system"] or \
               "Previous conversation" in second_call_kwargs["system"]


# ══════════════════════════════════════════════════════════════════════════════
# E. Integration — real ChromaDB + real search tool + mocked API
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationWithRealComponents:
    """
    Uses the real VectorStore, real CourseSearchTool, real SessionManager,
    and real AIGenerator — only the Anthropic HTTP client is mocked.
    Validates that no component interaction causes an uncaught exception.
    """

    @pytest.fixture
    def full_rag(self, tmp_path):
        from config import Config
        from ai_generator import AIGenerator
        from session_manager import SessionManager
        from vector_store import VectorStore
        from document_processor import DocumentProcessor
        from search_tools import CourseSearchTool, CourseOutlineTool

        cfg = Config(
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="test-model",
            CHROMA_PATH=str(tmp_path / "chroma"),
        )
        rag = object.__new__(RAGSystem)
        rag.config = cfg
        rag.document_processor = DocumentProcessor(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        rag.vector_store = VectorStore(cfg.CHROMA_PATH, cfg.EMBEDDING_MODEL, cfg.MAX_RESULTS)
        rag.session_manager = SessionManager(cfg.MAX_HISTORY)

        gen = object.__new__(AIGenerator)
        gen.model = cfg.ANTHROPIC_MODEL
        gen.base_params = {"model": cfg.ANTHROPIC_MODEL, "temperature": 0, "max_tokens": 800}
        gen.client = MagicMock()
        rag.ai_generator = gen

        rag.tool_manager = ToolManager()
        rag.search_tool = CourseSearchTool(rag.vector_store)
        rag.tool_manager.register_tool(rag.search_tool)
        rag.outline_tool = CourseOutlineTool(rag.vector_store)
        rag.tool_manager.register_tool(rag.outline_tool)

        return rag

    def _mock_tool_then_text(self, rag, tool_input, final_text):
        rag.ai_generator.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_int", tool_input),
            _direct_response(final_text),
        ]

    def test_content_query_on_empty_db_does_not_crash(self, full_rag):
        """
        An empty ChromaDB must not propagate an exception through the search tool.
        If this test fails, VectorStore.search() is not catching the ChromaDB error.
        """
        self._mock_tool_then_text(
            full_rag,
            {"query": "deep learning"},
            "No relevant content was found.",
        )
        answer, _ = full_rag.query("What is deep learning in the course?")
        assert isinstance(answer, str)

    def test_content_query_with_data_returns_answer(self, full_rag):
        chunks = [
            CourseChunk(
                content="Deep learning uses multi-layer neural networks for feature extraction.",
                course_title="DL 101",
                lesson_number=1,
                chunk_index=0,
            )
        ]
        full_rag.vector_store.add_course_content(chunks)
        self._mock_tool_then_text(
            full_rag,
            {"query": "deep learning neural networks"},
            "Deep learning uses multi-layer neural networks.",
        )
        answer, _ = full_rag.query("What is deep learning in the DL 101 course?")
        assert answer == "Deep learning uses multi-layer neural networks."

    def test_search_tool_sources_attached_to_response(self, full_rag):
        chunks = [
            CourseChunk(
                content="Transformers use self-attention mechanisms.",
                course_title="NLP Course",
                lesson_number=2,
                chunk_index=0,
            )
        ]
        full_rag.vector_store.add_course_content(chunks)
        self._mock_tool_then_text(
            full_rag,
            {"query": "transformers self-attention"},
            "Transformers use self-attention.",
        )
        _, sources = full_rag.query("Explain transformers from the NLP course")
        assert len(sources) > 0, (
            "Sources are empty after a real search returned content. "
            "Check that _format_results() populates last_sources and that "
            "ToolManager.get_last_sources() is called before reset_sources()."
        )

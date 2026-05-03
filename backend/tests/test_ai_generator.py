"""
Tests for AIGenerator in ai_generator.py.

Three concerns:
  1. Direct-response path — no tool call, just text back.
  2. Tool-use path — Claude responds with tool_use, the tool is executed,
     and the follow-up API call produces the final answer.
  3. Model validity — a real API call to confirm the configured model ID
     has not been deprecated (the most common silent breakage in this codebase).
"""

import pytest
from unittest.mock import MagicMock, call

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_generator(model="mock-model"):
    """Build an AIGenerator with a stubbed Anthropic client."""
    gen = object.__new__(AIGenerator)
    gen.client = MagicMock()
    gen.model = model
    gen.base_params = {"model": model, "temperature": 0, "max_tokens": 800}
    return gen


def _text_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _tool_use_response(name, tool_id, input_dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_dict
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def _search_manager(search_return="mock search results"):
    """ToolManager with a mock CourseSearchTool registered."""
    mock_tool = MagicMock(spec=CourseSearchTool)
    mock_tool.get_tool_definition.return_value = {
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    mock_tool.execute.return_value = search_return
    mock_tool.last_sources = []
    mgr = ToolManager()
    mgr.tools["search_course_content"] = mock_tool
    return mgr, mock_tool


# ══════════════════════════════════════════════════════════════════════════════
# 1. Direct-response path (no tool use)
# ══════════════════════════════════════════════════════════════════════════════

class TestDirectResponse:

    def test_returns_text_content(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("Paris.")
        assert gen.generate_response(query="Capital of France?") == "Paris."

    def test_system_prompt_included_in_api_call(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        gen.generate_response(query="hello")
        kwargs = gen.client.messages.create.call_args[1]
        assert "system" in kwargs and len(kwargs["system"]) > 0

    def test_conversation_history_appended_to_system(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        gen.generate_response(query="follow up", conversation_history="User: hi\nAssistant: hello")
        kwargs = gen.client.messages.create.call_args[1]
        assert "hi" in kwargs["system"] and "hello" in kwargs["system"]

    def test_tools_included_when_provided(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]
        gen.generate_response(query="q", tools=tools)
        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs.get("tools") == tools

    def test_tool_choice_auto_when_tools_provided(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        tools = [{"name": "t", "description": "d", "input_schema": {}}]
        gen.generate_response(query="q", tools=tools)
        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs.get("tool_choice") == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        gen.generate_response(query="q")
        kwargs = gen.client.messages.create.call_args[1]
        assert "tools" not in kwargs

    def test_only_one_api_call_on_direct_response(self):
        gen = _make_generator()
        gen.client.messages.create.return_value = _text_response("ok")
        gen.generate_response(query="q")
        assert gen.client.messages.create.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tool-use path — the critical path for content queries
# ══════════════════════════════════════════════════════════════════════════════

class TestToolUsePath:

    def test_search_tool_executed_when_claude_requests_it(self):
        gen = _make_generator()
        mgr, mock_tool = _search_manager("backprop info")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_01", {"query": "backpropagation"}),
            _text_response("Backpropagation is..."),
        ]
        gen.generate_response(
            query="Explain backpropagation",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        mock_tool.execute.assert_called_once_with(query="backpropagation")

    def test_final_answer_returned_after_tool_execution(self):
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_02", {"query": "q"}),
            _text_response("Final answer."),
        ]
        result = gen.generate_response(
            query="content question",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert result == "Final answer."

    def test_two_api_calls_made_on_tool_use(self):
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_03", {"query": "x"}),
            _text_response("done"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert gen.client.messages.create.call_count == 2

    def test_final_api_call_omits_tools(self):
        """The last API call (final synthesis after max rounds) must omit tools."""
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_04a", {"query": "x"}),
            _tool_use_response("search_course_content", "tu_04b", {"query": "y"}),
            _text_response("done"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        last_call_kwargs = gen.client.messages.create.call_args_list[-1][1]
        assert "tools" not in last_call_kwargs, (
            "Final API call must NOT include tools to prevent an infinite tool-use loop."
        )

    def test_tool_result_present_in_second_api_call_messages(self):
        gen = _make_generator()
        mgr, _ = _search_manager("found: gradient descent")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_05", {"query": "gradients"}),
            _text_response("Gradients are..."),
        ]
        gen.generate_response(
            query="gradients?",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        second_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        # Flatten all content blocks
        all_content = []
        for msg in second_msgs:
            c = msg.get("content")
            if isinstance(c, list):
                all_content.extend(c)
        tool_results = [b for b in all_content if isinstance(b, dict) and b.get("type") == "tool_result"]
        assert len(tool_results) == 1, "Exactly one tool_result block expected"
        assert tool_results[0]["content"] == "found: gradient descent"

    def test_tool_use_id_echoed_in_tool_result(self):
        gen = _make_generator()
        mgr, _ = _search_manager("result")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "MY_TOOL_ID", {"query": "x"}),
            _text_response("ok"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        second_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        all_content = []
        for msg in second_msgs:
            c = msg.get("content")
            if isinstance(c, list):
                all_content.extend(c)
        tool_results = [b for b in all_content if isinstance(b, dict) and b.get("type") == "tool_result"]
        assert tool_results[0]["tool_use_id"] == "MY_TOOL_ID"

    def test_assistant_tool_use_message_included(self):
        """The assistant's tool_use block must be in the message chain for the second call."""
        gen = _make_generator()
        mgr, _ = _search_manager()
        first_resp = _tool_use_response("search_course_content", "tu_06", {"query": "x"})
        gen.client.messages.create.side_effect = [first_resp, _text_response("ok")]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        second_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        assistant_msgs = [m for m in second_msgs if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        # The assistant message content must reference the original tool_use block
        assert assistant_msgs[0]["content"] is first_resp.content


# ══════════════════════════════════════════════════════════════════════════════
# 4. Two-round tool use — sequential chaining
# ══════════════════════════════════════════════════════════════════════════════

class TestTwoRoundToolUse:

    def test_two_tool_rounds_makes_three_api_calls(self):
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_r1", {"query": "outline"}),
            _tool_use_response("search_course_content", "tu_r2", {"query": "content"}),
            _text_response("Final answer."),
        ]
        gen.generate_response(
            query="complex query",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert gen.client.messages.create.call_count == 3

    def test_final_answer_returned_after_two_rounds(self):
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_r1", {"query": "q1"}),
            _tool_use_response("search_course_content", "tu_r2", {"query": "q2"}),
            _text_response("Answer."),
        ]
        result = gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert result == "Answer."

    def test_both_tools_executed_in_two_rounds(self):
        gen = _make_generator()
        mgr, mock_tool = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_r1", {"query": "round1"}),
            _tool_use_response("search_course_content", "tu_r2", {"query": "round2"}),
            _text_response("done"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert mock_tool.execute.call_count == 2

    def test_max_rounds_caps_at_two(self):
        """A third tool_use response must never be processed — loop caps at 2 rounds."""
        gen = _make_generator()
        mgr, mock_tool = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_r1", {"query": "q1"}),
            _tool_use_response("search_course_content", "tu_r2", {"query": "q2"}),
            _text_response("done"),
            _tool_use_response("search_course_content", "tu_r3", {"query": "q3"}),  # never reached
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert gen.client.messages.create.call_count == 3
        assert mock_tool.execute.call_count == 2

    def test_early_termination_no_extra_call(self):
        """One tool round followed by a direct text answer must make exactly 2 API calls."""
        gen = _make_generator()
        mgr, _ = _search_manager()
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_r1", {"query": "q"}),
            _text_response("Direct answer."),
        ]
        result = gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert gen.client.messages.create.call_count == 2
        assert result == "Direct answer."

    def test_tool_error_does_not_raise_continues_to_answer(self):
        """A tool execution exception must not propagate — loop continues to final answer."""
        gen = _make_generator()
        mgr, mock_tool = _search_manager()
        mock_tool.execute.side_effect = RuntimeError("DB unavailable")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "tu_err", {"query": "q"}),
            _text_response("Sorry, an error occurred."),
        ]
        result = gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        assert gen.client.messages.create.call_count == 2
        assert result == "Sorry, an error occurred."

    def test_tool_error_sent_as_is_error_block(self):
        """On tool execution error, an is_error tool_result block must be in the follow-up call."""
        gen = _make_generator()
        mgr, mock_tool = _search_manager()
        mock_tool.execute.side_effect = RuntimeError("timeout")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "err_id", {"query": "q"}),
            _text_response("ok"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        second_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        all_content = []
        for msg in second_msgs:
            c = msg.get("content")
            if isinstance(c, list):
                all_content.extend(c)
        error_results = [b for b in all_content if isinstance(b, dict) and b.get("is_error")]
        assert len(error_results) == 1
        assert error_results[0]["tool_use_id"] == "err_id"

    def test_two_round_message_chain_has_two_tool_results(self):
        """After two rounds, the final API call's messages must contain two tool_result blocks."""
        gen = _make_generator()
        mgr, _ = _search_manager("result data")
        gen.client.messages.create.side_effect = [
            _tool_use_response("search_course_content", "id_r1", {"query": "q1"}),
            _tool_use_response("search_course_content", "id_r2", {"query": "q2"}),
            _text_response("done"),
        ]
        gen.generate_response(
            query="q",
            tools=mgr.get_tool_definitions(),
            tool_manager=mgr,
        )
        final_msgs = gen.client.messages.create.call_args_list[-1][1]["messages"]
        all_content = []
        for msg in final_msgs:
            c = msg.get("content")
            if isinstance(c, list):
                all_content.extend(c)
        tool_results = [b for b in all_content if isinstance(b, dict) and b.get("type") == "tool_result"]
        assert len(tool_results) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model validity — real API call
# ══════════════════════════════════════════════════════════════════════════════

class TestModelValidity:
    """
    Makes an actual Anthropic API call with the configured model.

    If ANTHROPIC_MODEL in config.py names a deprecated or renamed model,
    this test fails with a clear message pointing to the fix.
    """

    @pytest.mark.integration
    def test_configured_model_accepts_requests(self):
        import anthropic
        from config import config

        if not config.ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not configured")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        try:
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply with one word: hi"}],
            )
            assert resp.content, "Model returned empty content"
            assert resp.content[0].text, "First content block has no text"
        except anthropic.NotFoundError as exc:
            pytest.fail(
                f"\n\nModel '{config.ANTHROPIC_MODEL}' was NOT FOUND.\n"
                f"It is likely deprecated. Update ANTHROPIC_MODEL in backend/config.py "
                f"to a current model ID (e.g. 'claude-sonnet-4-6').\n"
                f"Original error: {exc}"
            )
        except anthropic.BadRequestError as exc:
            pytest.fail(
                f"\n\nBad request for model '{config.ANTHROPIC_MODEL}': {exc}"
            )

    @pytest.mark.integration
    def test_tool_use_path_with_real_api(self):
        """
        Sends a course-specific question to the real API with the search tool
        available. Claude should call the tool; we verify the round-trip works
        end-to-end (tool_use → execute → final response).
        """
        import anthropic
        from config import config

        if not config.ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not configured")

        mgr, mock_tool = _search_manager("Mock lesson content: deep learning fundamentals.")
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        # First call — expect Claude to request the tool
        r1 = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=200,
            tools=mgr.get_tool_definitions(),
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": "What is covered in lesson 1 of the deep learning course?"}],
        )

        if r1.stop_reason != "tool_use":
            pytest.skip("Claude chose not to use a tool — cannot exercise the tool path")

        # Execute the tool
        tool_results = []
        for block in r1.content:
            if block.type == "tool_use":
                result = mgr.execute_tool(block.name, **block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        assert tool_results, "No tool was executed"

        # Second call — final answer
        r2 = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[
                {"role": "user", "content": "What is covered in lesson 1 of the deep learning course?"},
                {"role": "assistant", "content": r1.content},
                {"role": "user", "content": tool_results},
            ],
        )

        assert r2.content, "Final response has no content"
        assert r2.content[0].text, "Final response first block has no text"

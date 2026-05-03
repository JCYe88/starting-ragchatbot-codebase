import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **Outline or structure queries** ("what lessons does X cover?", "show me the outline of X", "what topics are in X?"): use `get_course_outline` — return the course title, course link, and every lesson number and title
- **Content or concept queries**: use `search_course_content` to find relevant material
- You may make up to 2 sequential tool calls when the first result is insufficient to fully answer the question. Use a second tool call only when the first result reveals a more specific query is needed (e.g., you retrieved a course outline and now need to search for related content). Do not use a second tool call by default.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using a tool
- **Course-specific questions**: Use the appropriate tool first, then answer
- **Outline responses**: Present the course title, course link, and a numbered list of all lessons (lesson number and title)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tool_manager)

        # Return direct response
        if not response.content:
            raise ValueError(f"Model returned empty content (stop_reason: {response.stop_reason})")
        return response.content[0].text

    def _execute_tool_calls(self, response, tool_manager) -> list:
        """Execute all tool_use blocks in a response, returning tool_result dicts."""
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                content = tool_manager.execute_tool(block.name, **block.input)
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": str(e),
                })
                continue
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        return results

    def _run_tool_loop(self, initial_response, base_params: Dict[str, Any], tool_manager) -> str:
        """
        Drive up to MAX_TOOL_ROUNDS of sequential tool calls, then return the final answer.

        Each round processes a tool_use response. Between rounds (when more rounds remain),
        a new API call with tools is made so Claude can chain another tool call. After all
        rounds are exhausted, one final no-tools call synthesizes the answer.

        Terminates early (without the final call) if Claude returns text between rounds.
        """
        messages = base_params["messages"].copy()
        system_content = base_params["system"]
        tools = base_params.get("tools", [])
        response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            tool_results = self._execute_tool_calls(response, tool_manager)
            messages.append({"role": "assistant", "content": response.content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Between rounds: make another with-tools call so Claude can chain
            if round_num < self.MAX_TOOL_ROUNDS - 1:
                response = self.client.messages.create(
                    **self.base_params,
                    messages=messages,
                    system=system_content,
                    tools=tools,
                    tool_choice={"type": "auto"},
                )
                if response.stop_reason != "tool_use":
                    # Claude answered directly — no final call needed
                    if not response.content:
                        raise ValueError(f"Model returned empty content (stop_reason: {response.stop_reason})")
                    return response.content[0].text

        # All rounds exhausted — final synthesis call without tools
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content,
        )
        if not final_response.content:
            raise ValueError(f"Model returned empty content after tool execution (stop_reason: {final_response.stop_reason})")
        return final_response.content[0].text
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with OpenAI's API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
1. **search_course_content** - Search within course materials for specific topics or concepts
2. **get_course_outline** - Get course structure including title, course link, and all lessons (number and title for each)

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure, syllabus, or table of contents
  - What lessons are in a course
  - Course overview or outline requests
  - When you need the course link or lesson list
  - Always include the course title, course link, and the number and title of each lesson in your response
- Use **search_course_content** for questions about:
  - Specific topics or concepts within course content
  - Detailed educational materials or explanations
- You may make up to 2 sequential tool calls if needed
- Use multiple searches when comparing courses or gathering related information
- Synthesize results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters (gpt-5 uses reasoning_effort instead of temperature)
        self.base_params = {
            "model": self.model,
            "reasoning_effort": "low",
            "max_completion_tokens": 800
        }

    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool definitions to OpenAI function format"""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

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

        # Build messages list for OpenAI
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }

        # Add tools if available (convert to OpenAI format)
        if tools:
            api_params["tools"] = self._convert_tools_to_openai_format(tools)
            api_params["tool_choice"] = "auto"

        # Get response from OpenAI
        response = self.client.chat.completions.create(**api_params)

        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            openai_tools = self._convert_tools_to_openai_format(tools) if tools else None
            return self._handle_tool_execution(response, messages, tool_manager, openai_tools)

        # Return direct response
        return response.choices[0].message.content

    def _handle_tool_execution(self, initial_response, messages: List[Dict], tool_manager, tools: Optional[List[Dict]] = None):
        """
        Handle execution of tool calls with support for sequential rounds.

        Args:
            initial_response: The response containing tool use requests
            messages: Current message list
            tool_manager: Manager to execute tools
            tools: OpenAI-formatted tool definitions for subsequent calls

        Returns:
            Final response text after tool execution
        """
        import json

        # Start with existing messages
        messages = messages.copy()
        current_response = initial_response
        round_count = 0

        while round_count < self.max_tool_rounds:
            assistant_message = current_response.choices[0].message

            # Add AI's tool use response
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute all tool calls and add results
            for tool_call in assistant_message.tool_calls:
                tool_args = json.loads(tool_call.function.arguments)
                try:
                    tool_result = tool_manager.execute_tool(
                        tool_call.function.name,
                        **tool_args
                    )
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            round_count += 1

            # Prepare API call with tools to allow another round
            api_params = {
                **self.base_params,
                "messages": messages
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # Get next response
            current_response = self.client.chat.completions.create(**api_params)

            # If no more tool calls, return the response
            if current_response.choices[0].finish_reason != "tool_calls":
                return current_response.choices[0].message.content

        # Max rounds reached - make final call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content

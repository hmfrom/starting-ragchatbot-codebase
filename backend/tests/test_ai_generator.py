"""Tests for AI generator."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


class TestAIGeneratorToolConversion:
    """Tests for tool format conversion."""

    def test_convert_tools_to_openai_format(self, mock_openai_client):
        """Test that tools are correctly converted to OpenAI format."""
        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }]

        converted = generator._convert_tools_to_openai_format(tools)

        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "search_course_content"
        assert converted[0]["function"]["description"] == "Search course materials"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_convert_multiple_tools(self, mock_openai_client):
        """Test conversion of multiple tools."""
        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]

        converted = generator._convert_tools_to_openai_format(tools)

        assert len(converted) == 2
        assert converted[0]["function"]["name"] == "tool1"
        assert converted[1]["function"]["name"] == "tool2"


class TestAIGeneratorResponse:
    """Tests for response generation."""

    def test_generate_response_without_tools(self, mock_openai_client):
        """Test basic response generation without tools."""
        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        response = generator.generate_response(query="What is Python?")

        assert response == "This is a test response."
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_generate_response_with_history(self, mock_openai_client):
        """Test response generation includes conversation history."""
        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        response = generator.generate_response(
            query="Follow up question",
            conversation_history="User: Hi\nAssistant: Hello!"
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "Previous conversation" in messages[0]["content"]

    def test_generate_response_passes_tools(self, mock_openai_client):
        """Test that tools are passed to the API when provided."""
        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        tools = [{
            "name": "search",
            "description": "Search tool",
            "input_schema": {"type": "object", "properties": {}}
        }]

        generator.generate_response(
            query="Search for something",
            tools=tools,
            tool_manager=Mock()
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tool_choice"] == "auto"


class TestAIGeneratorToolExecution:
    """Tests for tool execution handling."""

    def test_handle_tool_execution(self, mock_openai_client):
        """Test that tool calls are executed and results returned."""
        # Set up mock for tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = json.dumps({"query": "python basics"})

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        tool_response = Mock()
        tool_response.choices = [mock_choice]

        # Set up final response
        final_message = Mock()
        final_message.content = "Here are the Python basics..."

        final_choice = Mock()
        final_choice.message = final_message
        final_choice.finish_reason = "stop"

        final_response = Mock()
        final_response.choices = [final_choice]

        mock_openai_client.chat.completions.create.side_effect = [tool_response, final_response]

        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results for python basics"

        tools = [{
            "name": "search_course_content",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}}
        }]

        response = generator.generate_response(
            query="Tell me about Python",
            tools=tools,
            tool_manager=tool_manager
        )

        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python basics"
        )
        assert response == "Here are the Python basics..."

    def test_tool_execution_passes_correct_arguments(self, mock_openai_client):
        """Test that tool arguments are correctly parsed and passed."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = json.dumps({
            "query": "machine learning",
            "course_name": "AI Course",
            "lesson_number": 5
        })

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        tool_response = Mock()
        tool_response.choices = [mock_choice]

        final_message = Mock()
        final_message.content = "Final response"

        final_choice = Mock()
        final_choice.message = final_message

        final_response = Mock()
        final_response.choices = [final_choice]

        mock_openai_client.chat.completions.create.side_effect = [tool_response, final_response]

        with patch('ai_generator.OpenAI', return_value=mock_openai_client):
            generator = AIGenerator(api_key="test-key", model="gpt-4")

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Results"

        tools = [{
            "name": "search_course_content",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}}
        }]

        generator.generate_response(
            query="Find ML content",
            tools=tools,
            tool_manager=tool_manager
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning",
            course_name="AI Course",
            lesson_number=5
        )

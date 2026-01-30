"""Tests for search tools."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Tests for CourseSearchTool."""

    def test_execute_with_valid_results(self, mock_vector_store, sample_search_results):
        """Test search returns formatted results when content is found."""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="machine learning")

        assert "AI Fundamentals" in result
        assert "machine learning basics" in result
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test search returns appropriate message when no content is found."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test search passes course filter correctly."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="basics", course_name="Python")

        assert "in course 'Python'" in result
        mock_vector_store.search.assert_called_once_with(
            query="basics", course_name="Python", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test search passes lesson filter correctly."""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="basics", lesson_number=3)

        assert "in lesson 3" in result
        mock_vector_store.search.assert_called_once_with(
            query="basics", course_name=None, lesson_number=3
        )

    def test_execute_with_error(self, mock_vector_store, error_search_results):
        """Test search returns error message when vector store fails."""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert "Database connection failed" in result

    def test_last_sources_tracking(self, mock_vector_store, sample_search_results):
        """Test that last_sources is populated after search."""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="machine learning")

        assert len(tool.last_sources) == 2
        assert "AI Fundamentals" in tool.last_sources[0]

    def test_last_sources_with_links(self, mock_vector_store, sample_search_results):
        """Test that sources include links when available."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="machine learning")

        assert any(
            'href="https://example.com/lesson1"' in src for src in tool.last_sources
        )

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition is properly structured."""
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool."""

    def test_execute_returns_outline(self, mock_vector_store):
        """Test outline tool returns formatted course structure."""
        mock_vector_store.get_course_outline.return_value = {
            "title": "Python Basics",
            "course_link": "https://example.com/python",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Variables"},
            ],
        }
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Python")

        assert "Python Basics" in result
        assert "https://example.com/python" in result
        assert "Introduction" in result
        assert "Variables" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test outline tool handles missing course."""
        mock_vector_store.get_course_outline.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Nonexistent")

        assert "No course found" in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition is properly structured."""
        tool = CourseOutlineTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]


class TestToolManager:
    """Tests for ToolManager."""

    def test_register_and_execute_tool(self, mock_vector_store, sample_search_results):
        """Test registering and executing a tool."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(search_tool)
        result = manager.execute_tool("search_course_content", query="test")

        assert "AI Fundamentals" in result

    def test_execute_unknown_tool(self):
        """Test executing an unregistered tool returns error."""
        manager = ToolManager()

        result = manager.execute_tool("unknown_tool", arg="value")

        assert "Tool 'unknown_tool' not found" in result

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions."""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        manager.register_tool(CourseOutlineTool(mock_vector_store))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 2

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources clears all tool sources."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        sources = manager.get_last_sources()

        assert len(sources) == 0

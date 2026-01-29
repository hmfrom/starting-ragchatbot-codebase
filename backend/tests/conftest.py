"""Shared test fixtures for RAG chatbot tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


@pytest.fixture
def mock_search_results():
    """Factory fixture for creating SearchResults with test data."""
    def _create_results(
        documents: List[str] = None,
        metadata: List[Dict[str, Any]] = None,
        distances: List[float] = None,
        error: Optional[str] = None
    ) -> SearchResults:
        return SearchResults(
            documents=documents or [],
            metadata=metadata or [],
            distances=distances or [],
            error=error
        )
    return _create_results


@pytest.fixture
def sample_search_results(mock_search_results):
    """Sample search results with course content."""
    return mock_search_results(
        documents=[
            "This is content about machine learning basics.",
            "Neural networks are a key part of deep learning."
        ],
        metadata=[
            {"course_title": "AI Fundamentals", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "AI Fundamentals", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.2, 0.3]
    )


@pytest.fixture
def empty_search_results(mock_search_results):
    """Empty search results."""
    return mock_search_results()


@pytest.fixture
def error_search_results(mock_search_results):
    """Search results with error."""
    return mock_search_results(error="Database connection failed")


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing tools."""
    store = Mock()
    store.search = Mock()
    store.get_lesson_link = Mock(return_value=None)
    store.get_course_outline = Mock(return_value=None)
    return store


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing AI generator."""
    client = Mock()

    # Create a mock response structure
    mock_message = Mock()
    mock_message.content = "This is a test response."
    mock_message.tool_calls = None

    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = Mock()
    mock_response.choices = [mock_choice]

    client.chat.completions.create = Mock(return_value=mock_response)

    return client


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing."""
    manager = Mock()
    manager.get_tool_definitions = Mock(return_value=[])
    manager.execute_tool = Mock(return_value="Tool executed successfully")
    manager.get_last_sources = Mock(return_value=[])
    manager.reset_sources = Mock()
    return manager

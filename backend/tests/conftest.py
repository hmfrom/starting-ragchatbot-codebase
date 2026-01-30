"""Shared test fixtures for RAG chatbot tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


# ============================================================================
# API Test App and Fixtures
# ============================================================================
# We create a separate test app to avoid the static file mounting issue
# in the main app.py which requires frontend files that don't exist in tests.

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[str]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Create a FastAPI test app with mocked RAG system."""
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing."""
    rag = Mock()

    # Mock session manager
    rag.session_manager = Mock()
    rag.session_manager.create_session = Mock(return_value="test-session-123")

    # Mock query method
    rag.query = Mock(return_value=(
        "This is a test response about machine learning.",
        ["AI Fundamentals - Lesson 1", "AI Fundamentals - Lesson 2"]
    ))

    # Mock get_course_analytics
    rag.get_course_analytics = Mock(return_value={
        "total_courses": 3,
        "course_titles": ["AI Fundamentals", "Python Basics", "Data Science"]
    })

    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies."""
    return create_test_app(mock_rag_system)


@pytest.fixture
def client(test_app):
    """Create a TestClient for the test app."""
    return TestClient(test_app)


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

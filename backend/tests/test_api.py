"""Tests for FastAPI API endpoints."""

import pytest
from unittest.mock import Mock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint."""

    def test_query_with_valid_request(self, client, mock_rag_system):
        """Test successful query with valid request body."""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about machine learning."
        assert len(data["sources"]) == 2
        mock_rag_system.query.assert_called_once()

    def test_query_with_session_id(self, client, mock_rag_system):
        """Test query with existing session ID."""
        response = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        mock_rag_system.query.assert_called_with("Tell me more", "existing-session")

    def test_query_creates_session_when_not_provided(self, client, mock_rag_system):
        """Test that a new session is created when none provided."""
        response = client.post(
            "/api/query",
            json={"query": "Hello"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_empty_query(self, client):
        """Test query with empty string."""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        # Empty query is still valid, just returns empty results
        assert response.status_code == 200

    def test_query_missing_query_field(self, client):
        """Test query with missing required field."""
        response = client.post(
            "/api/query",
            json={}
        )
        assert response.status_code == 422  # Validation error

    def test_query_invalid_json(self, client):
        """Test query with invalid JSON."""
        response = client.post(
            "/api/query",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_query_handles_rag_system_error(self, client, mock_rag_system):
        """Test that RAG system errors return 500."""
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint."""

    def test_get_courses_returns_stats(self, client, mock_rag_system):
        """Test successful retrieval of course statistics."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "AI Fundamentals" in data["course_titles"]
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_empty_catalog(self, client, mock_rag_system):
        """Test response when no courses are loaded."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_handles_error(self, client, mock_rag_system):
        """Test that analytics errors return 500."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / root endpoint."""

    def test_root_returns_status(self, client):
        """Test root endpoint returns health check response."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestAPIValidation:
    """Tests for API request validation."""

    def test_query_with_extra_fields(self, client, mock_rag_system):
        """Test that extra fields in request are ignored."""
        response = client.post(
            "/api/query",
            json={
                "query": "Test query",
                "extra_field": "should be ignored",
                "another_extra": 123
            }
        )

        assert response.status_code == 200

    def test_query_with_null_session_id(self, client, mock_rag_system):
        """Test query with explicit null session_id."""
        response = client.post(
            "/api/query",
            json={"query": "Test", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_query_response_schema(self, client, mock_rag_system):
        """Test that response matches expected schema."""
        response = client.post(
            "/api/query",
            json={"query": "What is deep learning?"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify sources are strings
        for source in data["sources"]:
            assert isinstance(source, str)

    def test_courses_response_schema(self, client, mock_rag_system):
        """Test that courses response matches expected schema."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        for title in data["course_titles"]:
            assert isinstance(title, str)

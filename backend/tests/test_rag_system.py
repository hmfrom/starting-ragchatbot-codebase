"""Tests for RAG system integration."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    OPENAI_API_KEY: str = "test-key"
    OPENAI_MODEL: str = "gpt-4"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method."""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_returns_response_and_sources(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that query returns both response and sources."""
        from rag_system import RAGSystem

        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Test response about courses"
        mock_ai_generator_cls.return_value = mock_ai_generator

        mock_vector_store = Mock()
        mock_vector_store_cls.return_value = mock_vector_store

        mock_session_manager = Mock()
        mock_session_manager.get_conversation_history.return_value = None
        mock_session_manager_cls.return_value = mock_session_manager

        config = MockConfig()
        rag = RAGSystem(config)

        # Mock the tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(return_value=["Source 1", "Source 2"])
        rag.tool_manager.reset_sources = Mock()

        response, sources = rag.query("What is Python?")

        assert response == "Test response about courses"
        assert sources == ["Source 1", "Source 2"]

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_passes_tools_to_ai(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that query passes tool definitions to AI generator."""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Response"
        mock_ai_generator_cls.return_value = mock_ai_generator

        mock_vector_store_cls.return_value = Mock()
        mock_session_manager_cls.return_value = Mock()
        mock_session_manager_cls.return_value.get_conversation_history.return_value = (
            None
        )

        config = MockConfig()
        rag = RAGSystem(config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Test query")

        # Verify generate_response was called with tools
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_with_session_uses_history(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that query with session ID uses conversation history."""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Response"
        mock_ai_generator_cls.return_value = mock_ai_generator

        mock_vector_store_cls.return_value = Mock()

        mock_session_manager = Mock()
        mock_session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )
        mock_session_manager_cls.return_value = mock_session_manager

        config = MockConfig()
        rag = RAGSystem(config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Follow up question", session_id="session-123")

        # Verify history was passed
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "Previous conversation"

        # Verify exchange was added to history
        mock_session_manager.add_exchange.assert_called_once()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_resets_sources_after_retrieval(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that sources are reset after being retrieved."""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Response"
        mock_ai_generator_cls.return_value = mock_ai_generator

        mock_vector_store_cls.return_value = Mock()
        mock_session_manager_cls.return_value = Mock()
        mock_session_manager_cls.return_value.get_conversation_history.return_value = (
            None
        )

        config = MockConfig()
        rag = RAGSystem(config)
        rag.tool_manager.get_last_sources = Mock(return_value=["Source"])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Test")

        rag.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization."""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_vector_store_receives_max_results(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that VectorStore is initialized with MAX_RESULTS from config."""
        from rag_system import RAGSystem

        mock_ai_generator_cls.return_value = Mock()
        mock_vector_store_cls.return_value = Mock()
        mock_session_manager_cls.return_value = Mock()

        config = MockConfig(MAX_RESULTS=10)
        RAGSystem(config)

        # Verify VectorStore was called with max_results
        mock_vector_store_cls.assert_called_once_with(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, 10  # MAX_RESULTS
        )

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_zero_max_results_causes_empty_searches(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that MAX_RESULTS=0 would cause empty search results."""
        from rag_system import RAGSystem

        mock_ai_generator_cls.return_value = Mock()
        mock_vector_store_cls.return_value = Mock()
        mock_session_manager_cls.return_value = Mock()

        # This test documents the bug: MAX_RESULTS=0 causes no results
        config = MockConfig(MAX_RESULTS=0)
        RAGSystem(config)

        # VectorStore is initialized with max_results=0
        mock_vector_store_cls.assert_called_once_with(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, 0  # This is the bug!
        )


class TestRAGSystemToolRegistration:
    """Tests for tool registration in RAGSystem."""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_search_and_outline_tools_registered(
        self,
        mock_session_manager_cls,
        mock_doc_processor_cls,
        mock_vector_store_cls,
        mock_ai_generator_cls,
    ):
        """Test that both search and outline tools are registered."""
        from rag_system import RAGSystem

        mock_ai_generator_cls.return_value = Mock()
        mock_vector_store_cls.return_value = Mock()
        mock_session_manager_cls.return_value = Mock()

        config = MockConfig()
        rag = RAGSystem(config)

        definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [d["name"] for d in definitions]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

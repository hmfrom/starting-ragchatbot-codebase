"""Tests for configuration validation."""

import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from config import config


class TestConfig:
    """Test configuration settings."""

    def test_max_results_is_positive(self):
        """MAX_RESULTS must be greater than 0 to return search results."""
        assert config.MAX_RESULTS > 0, (
            f"MAX_RESULTS is {config.MAX_RESULTS}, but must be > 0 "
            "to return any search results from ChromaDB"
        )

    def test_max_results_reasonable_upper_bound(self):
        """MAX_RESULTS should not be excessively large."""
        assert config.MAX_RESULTS <= 20, (
            f"MAX_RESULTS is {config.MAX_RESULTS}, which may return too many results"
        )

    def test_chunk_size_is_positive(self):
        """CHUNK_SIZE must be positive for document processing."""
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        """CHUNK_OVERLAP must be less than CHUNK_SIZE."""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_max_history_is_non_negative(self):
        """MAX_HISTORY should be non-negative."""
        assert config.MAX_HISTORY >= 0

    def test_embedding_model_is_set(self):
        """EMBEDDING_MODEL must be configured."""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL must not be empty"

    def test_chroma_path_is_set(self):
        """CHROMA_PATH must be configured."""
        assert config.CHROMA_PATH, "CHROMA_PATH must not be empty"

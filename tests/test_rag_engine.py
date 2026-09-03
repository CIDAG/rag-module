import pytest
from src.core.interfaces import (
    BasePostProcessor, 
    BaseQueryProcessor,
    BaseRetriever,
)
from unittest.mock import Mock
from src.domain.rag_engine import RAGEngine
from src.core.entities import Document

class TestRAGEngineRouting:
    """
    Integration tests for the RAG pipeline routing.
    Ensures that components correctly pass data types along the junction points.
    """
    def test_retrieve_context_passes_correct_types_through_pipeline(self):
        mock_processor = Mock(spec=BaseQueryProcessor)
        mock_retriever = Mock(spec=BaseRetriever)
        mock_post_processor = Mock(spec=BasePostProcessor)

        dummy_doc = Document(
            page_content="Mocked content", 
            metadata={}, 
            score=0.9
        )
        mock_processor.process.return_value = ["processed query"]
        mock_retriever.retrieve.return_value = [dummy_doc]
        mock_post_processor.process.return_value = [dummy_doc]

        engine = RAGEngine(
            query_processor=mock_processor,
            retriever=mock_retriever,
            post_processor=mock_post_processor
        )

        raw_query = "original query"
        results = engine.retrieve_context(raw_query)
        mock_processor.process.assert_called_once_with(raw_query)
        mock_post_processor.process.assert_called_once_with([dummy_doc])

        assert results == [dummy_doc]
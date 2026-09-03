from src.core.interfaces import (
    BaseDocumentLoader, 
    BaseTextSplitter, 
    BaseDocumentSaver, 
    BaseRetriever,
    BaseEmbedder
)
from src.infrastructure.langchain_impl.loaders import LCLoader
from src.infrastructure.langchain_impl.splitters import LCSplitter
from src.infrastructure.vector_store.chroma_db import LCChromaVectorStore
from src.infrastructure.langchain_impl.embeddings import LCEmbedding


def test_loader_respects_contract() -> None:
    """Validates if LCLoader implements the BaseDocumentLoader interface."""
    assert issubclass(LCLoader, BaseDocumentLoader), (
        "LCLoader must inherit from BaseDocumentLoader"
    )

def test_splitter_respects_contract() -> None:
    """Validates if LCSplitter implements the BaseTextSplitter interface."""
    assert issubclass(LCSplitter, BaseTextSplitter), (
        "LCSplitter must inherit from BaseTextSplitter"
    )

def test_embedder_respects_contract() -> None:
    """Validates if LCEmbedding implements the BaseEmbedder interface."""
    assert issubclass(LCEmbedding, BaseEmbedder), (
        "LCEmbedding must inherit from BaseEmbedder"
    )

def test_vector_store_respects_multiple_contracts() -> None:
    """
    Validates Interface Segregation Principle (ISP). 
    
    The vector store must act as both Saver and Retriever, responding to two 
    distinct architectural contracts without mixing their scopes.
    """
    assert issubclass(LCChromaVectorStore, BaseDocumentSaver), (
        "LCChromaVectorStore must implement BaseDocumentSaver"
    )
    assert issubclass(LCChromaVectorStore, BaseRetriever), (
        "LCChromaVectorStore must implement BaseRetriever"
    )

def test_ingestion_engine_accepts_interfaces() -> None:
    """
    Tests if the IngestionEngine accepts any class based on the interfaces,
    proving low coupling and proper dependency injection.
    """
    from src.domain.ingestion_manager import IngestionEngine
    from unittest.mock import Mock
    
    # Create mock objects using Python's built-in library
    mock_loader = Mock(spec=BaseDocumentLoader)
    mock_splitter = Mock(spec=BaseTextSplitter)
    mock_saver = Mock(spec=BaseDocumentSaver)

    # If the engine initializes without typing errors, the injection contract works
    engine = IngestionEngine(
        loader=mock_loader,
        splitter=mock_splitter,
        saver=mock_saver
    )
    
    assert engine.loader is mock_loader
    assert engine.saver is mock_saver

def test_rag_engine_passes_list_to_retriever() -> None:
    """
    Ensures that the query is passed as a single-element list to the retriever,
    preventing regressions where raw strings are passed instead.
    """
    from unittest.mock import Mock
    from src.domain.rag_engine import RAGEngine
    from src.infrastructure.langchain_impl.processor import PassthroughQueryProcessor
    
    # 1. Arrange
    processor = PassthroughQueryProcessor() 
    
    mock_retriever = Mock()
    mock_post_processor = Mock()
    
    mock_retriever.retrieve.return_value = []
    mock_post_processor.process.return_value = []

    rag_engine = RAGEngine(
        query_processor=processor,
        retriever=mock_retriever,
        post_processor=mock_post_processor
    )

    query = "metrics"

    # 2. Act
    rag_engine.retrieve_context(query)

    # 3. Assert
    mock_retriever.retrieve.assert_called_once_with(["metrics"])
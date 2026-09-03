from pathlib import Path
from src.core.entities import Document
from src.infrastructure.langchain_impl.loaders import LCLoader 

def test_loader_finds_all_supported_extensions(
    temp_knowledge_dir: Path, 
    mock_langchain_loader_class: type
) -> None:
    """
    Ensures the load method recursively scans the directory and finds all supported extensions,
    ignoring unsupported file types.
    """
    loader = LCLoader()
    
    # Mock injection: Replace actual LangChain loaders with the mock class
    # to avoid reading real binaries and breaking the test environment.
    loader.SUPPORTED_EXTENSIONS = {
        "pdf": mock_langchain_loader_class,
        "json": mock_langchain_loader_class,
        "txt": mock_langchain_loader_class,
        "md": mock_langchain_loader_class,
        "xml": mock_langchain_loader_class,
    }

    docs = loader.load(temp_knowledge_dir)

    # Verify it found exactly 5 files, ignoring the unsupported .csv
    assert len(docs) == 5

def test_loader_translates_to_domain_entity(
    temp_knowledge_dir: Path, 
    mock_langchain_loader_class: type
) -> None:
    """
    Ensures that data read by third-party tools (LangChain) is properly 
    converted to the project's internal domain entity (Document).
    """
    loader = LCLoader()
    loader.SUPPORTED_EXTENSIONS = {"txt": mock_langchain_loader_class}

    txt_file = temp_knowledge_dir / "notes.txt"
    docs = loader._TXT_load(txt_file)

    assert len(docs) == 1
    
    # Core domain translation validation
    assert isinstance(docs[0], Document)
    assert docs[0].score == 0.0  # During ingestion, the score must be strictly zero
    assert "Fake content" in docs[0].page_content
    assert "source" in docs[0].metadata
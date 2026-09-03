import pytest
from pathlib import Path
from src.core.entities import Document

class MockLangChainLoader:
    """
    A mock class to replace actual LangChain loaders during tests.
    
    This prevents the test suite from requiring heavy, real binary files (like PDFs), 
    returning an object structurally identical to what LangChain yields.
    """
    def __init__(self, file_path: Path | str, **kwargs) -> None:
        self.file_path = file_path

    def load(self) -> list:
        class DummyLangChainDoc:
            def __init__(self, path: Path | str) -> None:
                self.page_content = f"Fake content for file {path}"
                self.metadata = {"source": str(path)}
                
        return [DummyLangChainDoc(self.file_path)]


@pytest.fixture
def mock_langchain_loader_class() -> type:
    """
    Fixture that provides the mock loader class.

    Returns:
        type: The MockLangChainLoader class reference.
    """
    return MockLangChainLoader


@pytest.fixture
def temp_knowledge_dir():
    """
    Creates an isolated temporary directory with empty files for each supported extension.
    """
    base_dir = Path(__file__).parent.parent
    mock_dir = base_dir / "data" / "tests"
    
    mock_dir.mkdir(parents=True, exist_ok=True)
    mock_files = [
        mock_dir / "article.pdf",
        mock_dir / "data.json",
        mock_dir / "notes.txt",
        mock_dir / "readme.md",
        mock_dir / "config.xml",
        mock_dir / "ignored_file.csv"
    ]
    
    for file in mock_files:
        file.touch()
    
    yield mock_dir
    
    for file in mock_files:
        if file.exists():
            file.unlink()


@pytest.fixture
def dummy_document() -> Document:
    """
    Provides a standard domain Document instance for pipeline testing.

    Returns:
        Document: A dummy document entity.
    """
    return Document(
        page_content="Test string for embeddings and splitters.",
        metadata={"source": "test.txt"},
        score=0.0
    )
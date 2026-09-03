from pathlib import Path
from src.infrastructure.langchain_impl.splitters import LCSplitter
from src.core.entities import Document

def test_splitter_breaks_long_text_and_preserves_metadata() -> None:
    """
    Ensures that the LCSplitter correctly divides a real, long text into 
    multiple chunks while preserving the original metadata across all chunks.
    """
    long_text_path = Path(__file__).parent.parent / "data" / "raw_articles" / "sample" / "mock_text.txt"
    
    with open(long_text_path, "r", encoding="utf-8") as file:
        massive_text = file.read()
    
    # Mock a raw Document exactly as it would be returned by the Loader
    mock_document = Document(
        page_content=massive_text,
        metadata={"source": str(long_text_path), "author": "Tester"},
        score=0.0
    )
    
    splitter = LCSplitter()
    chunks = splitter.split_text([mock_document])
    
    assert len(chunks) > 1, (
        "The splitter failed to divide the long text into multiple chunks."
    )
    
    # Verify maximum size constraint
    for chunk in chunks:
        assert len(chunk.page_content) <= splitter.chunk_size, (
            "A chunk exceeded the maximum character limit."
        )
        
    # Verify exact metadata preservation on the first and last chunks
    assert chunks[0].metadata["source"] == str(long_text_path), (
        "Source metadata was lost in the first chunk."
    )
    assert chunks[-1].metadata["author"] == "Tester", (
        "Author metadata was lost in the last chunk."
    )
    assert chunks[0].score == 0.0, (
        "Score was improperly modified during splitting."
    )
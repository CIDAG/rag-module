from pathlib import Path
from src.core.interfaces import BaseDocumentLoader, BaseTextSplitter, BaseDocumentSaver
from src.core.entities import Document

class IngestionEngine:
    def __init__(
            self,
            loader: BaseDocumentLoader,
            splitter: BaseTextSplitter,
            saver: BaseDocumentSaver
    ) -> None:
        """
        Initialize the Ingestion Engine with specific pipeline components.

        Args:
            loader (BaseDocumentLoader): Component to read raw files into Document objects.
            splitter (BaseTextSplitter): Component to divide large documents into smaller chunks.
            saver (BaseDocumentSaver): Component to persist chunks into a vector store.
        """
        self.loader = loader
        self.splitter = splitter
        self.saver = saver

    def _generate_chunks_ids(self, chunks: list[Document]) -> list[str]:
        import hashlib

        ids_list = []
        for chunk in chunks:
            unique_str = f"{chunk.page_content}{chunk.metadata.get('source')}{chunk.metadata.get('page')}"
            hash_id = hashlib.sha256(unique_str.encode('utf-8')).hexdigest()
            ids_list.append(hash_id)
        return ids_list

    def ingest_content(self, path: Path | str) -> None:
        """
        Execute the full ingestion pipeline for a given path.

        This method loads documents from the specified file or directory path,
        splits them into smaller chunks, and saves those chunks to the vector database.

        Args:
            path (Path | str): The file or directory path containing the raw documents.

        Returns:
            None

        Example:
            >>> from pathlib import Path
            >>> engine = IngestionEngine(my_loader, my_splitter, my_saver)
            >>> docs_path = Path("data/raw_articles/sample.pdf")
            >>> engine.ingest_content(docs_path)
        """
        loaded_docs = self.loader.load(path)
        chunks = self.splitter.split_text(loaded_docs)
        chunks_ids = self._generate_chunks_ids(chunks)
        self.saver.save(docs=chunks, hash_ids=chunks_ids)
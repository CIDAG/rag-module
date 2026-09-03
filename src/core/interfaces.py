from abc import ABC, abstractmethod
from pathlib import Path
from src.core.entities import Document
from typing import Any

################################ RAG INTERFACES ################################
class BaseQueryProcessor(ABC):
    """
    Query processing contract.
    """
    @abstractmethod
    def process(self, query: str) -> list[str]:
        """
        Receive a **query** and apply a rewriting or expansion method.

        Args:
            query (str): User input.
        Returns:
            list[str]: Expanded or rewritten queriy.

        Example:
            >>> processor = SpecificQueryProcessor()
            >>> processor.process("chart metrics")
            ['chart metrics', 'what are the metrics of the chart?']
        """
        pass

class BaseRetriever(ABC):
    """
    Retrieving base contract.from pathlib import Path
    """
    @abstractmethod
    def retrieve(self, queries: list[str]) -> list[Document]:
        """
        Use processed queries to search the vector database and retrieve relevant data.

        Args:
            queries (list[str]): Processed user inputs or expanded queries.

        Returns:
            list[Document]: Retrieved article chunks.

        Example:
            >>> retriever = SpecificRetriever()
            >>> docs = retriever.retrieve(["chart metrics"])
            >>> print(docs[0].page_content)
            'The evaluation metrics for this chart include...'
        """
        pass

class BasePostProcessor(ABC):
    """
    Post-processing contract for filtering or re-ranking.
    """
    @abstractmethod
    def process(self, docs: list[Document]) -> list[Document]:
        """
        Apply a filter, transformation or sorting to the retrieved documents.

        Args:
            docs (list[Document]): Retrieved data from the vector database.

        Returns:
            list[Document]: Altered or filtered document ranking.

        Example:
            >>> processor = SpecificPostProcessor()
            >>> refined_docs = processor.process(raw_docs)
            >>> len(refined_docs) <= len(raw_docs)
            True
        """
        pass

class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models.
    """
    @abstractmethod
    def embed_documents(self, docs: list[Document]) -> list[list[float]]:
        """
        Converts a list of Document entities into their corresponding vector representations.

        Args:
            docs (list[Document]): A list of Document entities containing the text 
                chunks to be vectorized.

        Returns:
            list[list[float]]: A list of embeddings, where each element is a list 
            of floats representing the multi-dimensional vector for a specific document chunk.

        Example:
            >>> # Assuming 'SpecificEmbedder' is a concrete implementation of this interface
            >>> embedder = SpecificEmbedder()
            >>> doc1 = Document(page_content="Machine learning is fascinating.", metadata={}, score=0.0)
            >>> doc2 = Document(page_content="Vector databases are fast.", metadata={}, score=0.0)
            >>> vectors = embedder.embed_documents([doc1, doc2])
            >>> len(vectors)
            2
            >>> isinstance(vectors[0][0], float)
            True
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Converts a single text string into its vector representation.

        Args:
            query (str): The search query or text input to be vectorized.

        Returns:
            list[float]: A list of floats representing the multi-dimensional 
            vector for the query.

        Example:
            >>> # Assuming 'SpecificEmbedder' is a concrete implementation of this interface
            >>> embedder = SpecificEmbedder()
            >>> query_vector = embedder.embed_query("What is a neural network?")
            >>> isinstance(query_vector, list)
            True
            >>> isinstance(query_vector[0], float)
            True
            >>> len(query_vector) # Will vary based on the model (e.g., 384 for all-MiniLM-L6-v2)
            384
        """
        pass

    @abstractmethod
    def get_model(self) -> Any:
        pass
################################################################################

########################## VECTOR DATABASE INTERFACES ##########################
class BaseDocumentLoader(ABC):
    """
    Document loading contract.
    """
    @abstractmethod
    def load(self, path: Path | str) -> list[Document]:
        """
        Process raw documents into Document objects.

        Args:
            path (Path | str): Articles database path

        Returns:
            list[Document]: Processed Document objects.

        Example:
            >>> docs_path = Path("PATH")
            >>> docs_loader = SpecificDocumentLoader()
            >>> loaded_docs = docs_loader.load(docs_path)
            >>> print(loaded_docs[0].page_content)
            'The evaluation metrics for this chart include...'
        """
        pass

class BaseTextSplitter(ABC):
    """
    Contract for splitting text into chunks.
    """
    @abstractmethod
    def split_text(self, docs: list[Document]) -> list[Document]:
        """
        Split text into chunks for storing/retrieving purposes.

        Args:
            docs (list[Document]): Raw Document objects.

        Returns:
            list[Document]: Chunked data.

        Example:
            >>> splitter = SpecificTextSplitter()
            >>> raw_docs = [Document(page_content="A very long text that needs splitting...", metadata={})]
            >>> chunks = splitter.split_text(raw_docs)
            >>> len(chunks) > len(raw_docs)
            True
        """
        pass

class BaseDocumentSaver(ABC):
    """
    Contract for saving documents to a vector database.
    """
    @abstractmethod
    def save(self, docs: list[Document], hash_ids: list[str]) -> None:
        """
        Persist a list of documents into the underlying vector store.

        Args:
            docs (list[Document]): The processed and chunked documents to save.
            hash_id (list[str]): Indexing ids relative to each chunk.

        Returns:
            None

        Example:
            >>> saver = SpecificDocumentSaver()
            >>> new_docs = [Document(page_content="Data", metadata={"source": "file.pdf"})]
            >>> ids = IdGenerator().generate_ids(new_docs)
            >>> saver.save(new_docs, ids)
        """
        pass
################################################################################
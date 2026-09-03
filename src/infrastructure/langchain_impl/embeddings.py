from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.core.interfaces import BaseEmbedder
from src.core.entities import Document
from typing import Any

class EmbeddingFactory:
    """
    A factory class responsible for instantiating the correct LangChain 
    embedding model based on the requested AI provider.
    """
    @staticmethod
    def initialize_model(
        provider: str,
        model_name: str,
        **kwargs
    ) -> Embeddings:
        """
        Initializes and returns a specific LangChain embedding model.

        Args:
            provider (str): The name of the AI provider (e.g., 'huggingface', 'openai').
            model_name (str): The specific model identifier for the provider.
            **kwargs: Additional keyword arguments passed directly to the model's constructor.

        Returns:
            Embeddings: An instantiated LangChain Embeddings object.

        Raises:
            ValueError: If the provided `provider` string is not supported by the factory.

        Example:
            >>> model = EmbeddingFactory.initialize_model("huggingface", "all-MiniLM-L6-v2")
            >>> type(model).__name__
            'HuggingFaceEmbeddings'
        """
        if provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        elif provider == "openai":
            # Requires OpenAI environment key (OPENAI_API_KEY)
            return OpenAIEmbeddings(model=model_name, **kwargs)
        else:
            raise ValueError(f"Embedding provider not supported: {provider}")

class LCEmbedding(BaseEmbedder):
    """
    Adapter class for LangChain embedding models using a Factory pattern.
    """
    def __init__(
            self,
            provider: str,
            model_name: str,
            **kwargs
    ):
        """Initializes the LCEmbedding adapter with the specified provider and model."""
        self.embedding_model: Embeddings = EmbeddingFactory.initialize_model(provider, model_name, **kwargs)

    def embed_documents(self, docs: list[Document]) -> list[list[float]]:
        """
        Converts a list of Document entities into their corresponding vector representations.

        Args:
            docs (list[Document]): A list of Document entities containing the text chunks.

        Returns:
            list[list[float]]: A matrix of embeddings, where each inner list represents 
            the vector for a specific document chunk.

        Example:
            >>> embedder = LCEmbedding()
            >>> doc1 = Document(page_content="Data analysis is crucial.", metadata={}, score=0.0)
            >>> vectors = embedder.embed_documents([doc1])
            >>> len(vectors)
            1
            >>> isinstance(vectors[0][0], float)
            True
        """
        texts = [doc.page_content for doc in docs]
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Converts a single text string into its vector representation.

        Args:
            query (str): The search query to be vectorized.

        Returns:
            list[float]: A list of floats representing the vector for the query.
        
        Example:
            >>> embedder = LCEmbedding()
            >>> vector = embedder.embed_query("What is a neural network?")
            >>> isinstance(vector, list)
            True
            >>> isinstance(vector[0], float)
            True
        """
        return self.embedding_model.embed_query(query)

    def get_model(self) -> Any:
        """
        Returns the raw LangChain embedding model instance.

        Returns:
            Any: The underlying LangChain model object (e.g., HuggingFaceEmbeddings),
            necessary for initializing certain LangChain vector stores like Chroma.
        
        Example:
            >>> embedder = LCEmbedding(provider="huggingface")
            >>> raw_model = embedder.get_model()
            >>> type(raw_model).__name__
            'HuggingFaceEmbeddings'
        """
        return self.embedding_model
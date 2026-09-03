from pathlib import Path
from typing import Any
from src.core.entities import Document
from src.core.interfaces import BaseDocumentSaver, BaseRetriever, BaseEmbedder

class LCChromaVectorStore(BaseDocumentSaver, BaseRetriever):
    """
    LangChain ChromaDB implementation for saving and retrieving vectors.
    """
    def __init__(self,
                 persist_path: Path | str,
                 embedding_model: BaseEmbedder
    ) -> None:
        """
        Initialize the Chroma vector store connection.

        Args:
            persist_path (Path | str): Path to the local directory where the database will be saved.
            embedding_model (Any): The embedding model instance used to translate text into vectors.
        """
        from langchain_chroma import Chroma

        self.persist_path = str(persist_path) 
        self.embedding_model = embedding_model.get_model()

        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_path
        )

    def save(self, docs: list[Document], hash_ids: list[str]) -> None:
        """
        Translate domain Documents to LangChain format and persist them into the vector database.

        Args:
            docs (list[Document]): The processed and chunked documents to save.
            hash_id (list[str]): Indexing ids relative to each chunk.
            
        Returns:
            None

        Example:
            >>> vector_store = ChromaVectorStore("./data", embedding_model)
            >>> new_docs = [Document(page_content="Data", metadata={"source": "file.pdf"})]
            >>> ids = IdGenerator().generate_ids(new_docs)
            >>> vector_store.save(new_docs, ids)
        """
        from langchain_core.documents import Document as LangchainDocument
        # Firstly, we need to translate our Document structure to LangChain Document
        lang_chain_docs = [
            LangchainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
            ) for doc in docs 
        ]
        # Then, we save it
        self.vector_store.add_documents(documents=lang_chain_docs, ids=hash_ids)

    def retrieve(self, queries: list[str]) -> list[Document]:
        """
        Perform similarity search using multiple queries, filtering out duplicate text contents.

        Args:
            queries (list[str]): Processed user inputs or expanded queries.

        Returns:
            list[Document]: Unique retrieved article chunks mapped to our domain entity, including similarity scores.

        Example:
            >>> vector_store = ChromaVectorStore("./data", embedding_model)
            >>> docs = vector_store.retrieve(["chart metrics", "data evaluation"])
            >>> print(docs[0].score)
            0.154
        """
        results = []
        retrieved = set()

        for query in queries:
            searches = self.vector_store.similarity_search_with_score(query)

            for chunk, score in searches:
                if chunk.page_content not in retrieved:
                    retrieved.add(chunk.page_content)

                    doc = Document(
                        page_content=chunk.page_content,
                        metadata={**chunk.metadata},
                        score=score
                    )
                    results.append(doc)
        
        return results
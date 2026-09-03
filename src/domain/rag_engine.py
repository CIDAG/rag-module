from src.core.entities import Document
from src.core.interfaces import BaseQueryProcessor, BaseRetriever, BasePostProcessor

class RAGEngine:
    def __init__(
            self,
            query_processor: BaseQueryProcessor,
            retriever: BaseRetriever,
            post_processor: BasePostProcessor
    ) -> None:
        """
        Initialize the RAG Engine with specific pipeline components.

        Args:
            query_processor (BaseQueryProcessor): Component to rewrite or expand queries.
            retriever (BaseRetriever): Component to fetch documents from the vector store.
            post_processor (BasePostProcessor): Component to filter or rank the documents.
        """
        self.query_processor = query_processor
        self.retriever = retriever
        self.post_processor = post_processor

    def retrieve_context(self, query: str) -> list[Document]:
        """
        Retrieve content based on user query.

        Args:
            query (str): User input
        
        Returns:
            list[Document]: Retrieved article chunks.

        Example:
            >>> retriever = RAGEngine(my_processor, my_retriever, my_ranker)
            >>> docs = retriever.retrieve_context("chart metrics")
            >>> print(docs[0].page_content)
            'The evaluation metrics for this chart include...'
        """
        processed_query = self.query_processor.process(query)
        document_list = self.retriever.retrieve(processed_query)
        refined_document_list = self.post_processor.process(document_list)

        return refined_document_list
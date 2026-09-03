from src.core.interfaces import BaseQueryProcessor, BasePostProcessor
from src.core.entities import Document

#############################################################################################
"""
Since LangChain already abstracts the processing step, we must define an empty processor
to fulfill the facade parameters.
"""
class PassthroughQueryProcessor(BaseQueryProcessor):
    """
    A null object implementation for query processing.
    
    This class acts as a passthrough, returning the user query exactly
    as it was received. It is useful for fulfilling architectural contracts 
    when no active query rewriting or expansion is needed.
    """
    def process(self, query: str) -> list[str]:
        """
        Return the input query unmodified.

        Args:
            query (str): The original user input or query.

        Returns:
            list[str]: The exact same unmodified query at a list format.

        Example:
            >>> processor = PassthroughQueryProcessor()
            >>> processed_query = processor.process("What are RAG metrics?")
            >>> print(processed_query)
            'What are RAG metrics?'
        """
        return [query]

class PassthroughPostProcessor(BasePostProcessor):
    """
    A null object implementation for document post-processing.
    
    This class acts as a passthrough, returning the retrieved documents 
    exactly as they were received. It fulfills the architectural contract 
    when no active reranking or filtering is required in the pipeline.
    """
    def process(self, documents: list[Document]) -> list[Document]:
        """
        Return the retrieved documents unmodified.

        Args:
            documents (list[Document]): The initial list of retrieved documents.

        Returns:
            list[Document]: The exact same unmodified list of documents.

        Example:
            >>> post_processor = PassthroughPostProcessor()
            >>> raw_docs = [Document(content="Some text", metadata={}, score=0.9)]
            >>> refined_docs = post_processor.process(raw_docs)
            >>> print(refined_docs == raw_docs)
            True
        """
        return documents
#############################################################################################
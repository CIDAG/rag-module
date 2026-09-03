from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.entities import Document
from src.core.interfaces import BaseTextSplitter

class LCSplitter(BaseTextSplitter):
    """
    Splits documents into smaller chunks for optimal LLM processing and vector search.
    """
    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 100
    ):
        """
        Initializes the LCSplitter with standard chunking configurations.
        
        Sets up the RecursiveCharacterTextSplitter with a chunk_size of 500 
        and a chunk_overlap of 100 characters for the MVP.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self, docs: list[Document]) -> list[Document]:
        """
        Splits a list of Document entities into smaller, chunked Document entities.

        Args:
            docs: A list of Document entities to be processed.

        Returns:
            list[Document]: A flat list of new Document entities, where each 
            contains a specific text chunk and inherits the original metadata.

        Example:
            >>> long_doc = Document(page_content="A very long text...", metadata={"source": "test.txt"}, score=0.0)
            >>> splitter = LCSplitter()
            >>> chunked_docs = splitter.split_text([long_doc])
            >>> print(chunked_docs[0].metadata["source"])
            'test.txt'
        """
        docs_list = []
        for doc in docs:
            splitted_strings = self.text_splitter.split_text(doc.page_content)

            for string in splitted_strings:
                splitted_doc = Document(
                    page_content=string,
                    metadata=doc.metadata,
                    score=0.0
                )
                docs_list.append(splitted_doc)

        return docs_list
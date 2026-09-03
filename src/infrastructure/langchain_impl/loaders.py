from pathlib import Path
from src.core.entities import Document
from src.core.interfaces import BaseDocumentLoader

class LCLoader(BaseDocumentLoader):
    """
    Loads documents from a directory or a single file, automatically
    routing to the correct LangChain loader based on file extension.
    """
    def __init__(self) -> None:
        """Initializes the LCLoader with supported file extensions."""
        from langchain_community.document_loaders import JSONLoader, PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredXMLLoader
        self.SUPPORTED_EXTENSIONS = {
            "pdf": PyMuPDFLoader,
            "json": JSONLoader,
            "txt": TextLoader,
            "md": UnstructuredMarkdownLoader,
            "xml": UnstructuredXMLLoader,
        }

    def _PDF_load(self, file: Path | str) -> list[Document]:
        """
        Loads and parses a PDF file into a list of Document entities.

        Args:
            file (Path | str): The path to the PDF file.

        Returns:
            list[Document]: A list containing a Document object for each page.
        
        Example:
            >>> loader = LCLoader()
            >>> docs = loader._PDF_load("article.pdf")
            >>> isinstance(docs, list)
            True
            >>> isinstance(docs[0], Document)
            True
        """
        loader = self.SUPPORTED_EXTENSIONS["pdf"](file)
        loaded_doc = loader.load()

        pages = []
        # Since the load() method stores the pdf pages on each list index,
        # we need to iterate over it 
        for page in loaded_doc:
            formated_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata,
                score=0.0
            )
            pages.append(formated_doc)

        return pages
    
    def _JSON_load(self, file: Path | str) -> list[Document]:
        """
        Loads and parses a JSON file into a list of Document entities.

        Args:
            file (Path | str): The path to the JSON file.

        Returns:
            list[Document]: A list of Document objects extracted from the JSON.
        
        Example:
            >>> loader = LCLoader()
            >>> docs = loader._JSON_load("data.json")
            >>> hasattr(docs[0], "page_content")
            True
        """
        loader = self.SUPPORTED_EXTENSIONS["json"](
            file_path=file,
            jq_schema=".[].text", # may cause errors depending on the json keys
            text_content=False
        )
        loaded_doc = loader.load()
        #print(loaded_doc[0].page_content)
        pages = []
        for page in loaded_doc:
            formated_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata,
                score=0.0
            )
            pages.append(formated_doc)
        
        return pages

    def _TXT_load(self, file: Path | str) -> list[Document]:
        """
        Loads and parses a text file with encoding fallback.

        Attempts to load the file using UTF-8. If a RuntimeError occurs 
        (LangChain's wrapper for UnicodeDecodeError), it falls back to ISO-8859-1.

        Args:
            file (Path | str): The path to the text file.

        Returns:
            list[Document]: A list containing the parsed text Document.
        
        Example:
            >>> loader = LCLoader()
            >>> docs = loader._TXT_load("notes.txt")
            >>> docs[0].score == 0.0
            True
        """
        # Due to LangChain dependencies errors, we can have trouble using the
        # autodetect_encoding parameter. As it is, we try forcing iso 8859-1 if
        # utf-8 doesn't load.
        try:
            loader = self.SUPPORTED_EXTENSIONS["txt"](
                file_path=file,
                encoding="utf-8"
            )
            loaded_doc = loader.load()
        except RuntimeError: # since LangChain catches UnicodeDecodeError, we have to catch RuntimeError
            loader = self.SUPPORTED_EXTENSIONS["txt"](
                file_path=file,
                encoding="iso-8859-1"
            )
            loaded_doc = loader.load()

        #print(loaded_doc[0].page_content)
        pages = []
        for page in loaded_doc:
            formated_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata,
                score=0.0
            )
            pages.append(formated_doc)
        
        return pages

    def _MD_load(self, file: Path | str) -> list[Document]:
        """
        Loads and parses a Markdown file into Document entities.

        Args:
            file (Path | str): The path to the Markdown file.

        Returns:
            list[Document]: A list containing the parsed Markdown Document.
        
        Example:
            >>> loader = LCLoader()
            >>> docs = loader._MD_load("readme.md")
            >>> isinstance(docs[0].page_content, str)
            True
        """
        loader = self.SUPPORTED_EXTENSIONS["md"](
            file_path=file,
            strategy="fast"
        )
        loaded_doc = loader.load()
        #print(loaded_doc[0].page_content)
        pages = []
        for page in loaded_doc:
            formated_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata,
                score=0.0
            )
            pages.append(formated_doc)
        
        return pages

    def _XML_load(self, file: Path | str) -> list[Document]:
        """
        Loads and parses a XML file into Document entities.

        Args:
            file (Path | str): The path to the XML file.

        Returns:
            list[Document]: A list containing the parsed XML Document.
        
        Example:
            >>> loader = LCLoader()
            >>> docs = loader._XML_load("config.xml")
            >>> isinstance(docs[0].metadata, dict)
            True
        """
        loader = self.SUPPORTED_EXTENSIONS["xml"](
            file_path=file,
            strategy="fast"
        )
        loaded_doc = loader.load()
        #print(loaded_doc[0].page_content)
        pages = []
        for page in loaded_doc:
            formated_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata,
                score=0.0
            )
            pages.append(formated_doc)
        
        return pages

    def load(self, path: Path | str) -> list[Document]:
        """
        Recursively searches for supported files in a directory and loads them.

        Args:
            path (Path | str): The target directory or file path.

        Returns:
            list[Document]: A flat list of all loaded Document entities.

        Example:
            >>> loader = LCLoader()
            >>> docs = loader.load(Path("data/raw_articles"))
            >>> print(f"Loaded {len(docs)} pages.")
            Loaded 42 pages.
            >>> type(docs)
            <class 'list'>
        """
        # We make sure a Path object is created
        target_path = Path(path)
        docs = []

        for file in target_path.rglob("*.pdf"):
            pdf_doc = self._PDF_load(file)
            docs.extend(pdf_doc)

        for file in target_path.rglob("*.json"):
            json_doc = self._JSON_load(file)
            docs.extend(json_doc)

        for file in target_path.rglob("*.txt"):
            txt_doc = self._TXT_load(file)
            docs.extend(txt_doc)

        for file in target_path.rglob("*.md"):
            md_doc = self._MD_load(file)
            docs.extend(md_doc)

        for file in target_path.rglob("*.xml"):
            xml_doc = self._XML_load(file)
            docs.extend(xml_doc)

        return docs
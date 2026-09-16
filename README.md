# Knowledge Module - Retrieval-Augmented Generation (RAG)

Modular Retrieval-Augmented Generation (RAG) system for the **Chart Analysis Framework**. As Part 3 of the platform, this module is responsible for recovering domain knowledge and contextual data to feed the Reasoning Module.

## Features

* **Clean Architecture Principles**: Strictly decoupled domain engines and infrastructural implementations via stable interfaces.
* **Deterministic Ingestion**: SHA-256 chunk hashing (combining text, source, and page metadata) to completely prevent vector database duplication.
* **Multi-Format Support**: Native parsing for PDF, JSON, TXT, MD, and XML files.
* **Configurable Embeddings**: Easily swap between HuggingFace and OpenAI embedding providers.
* **Visual Demo**: Built-in Streamlit interface for isolated testing and debugging.

## Architecture

Following the monorepo structure, this module separates the core business logic from third-party tools like LangChain and ChromaDB:

```
knowledge_module/
├── config/
│   ├── config.py                 # Modular configuration loader
│   └── models.yaml               # Embeddings and splitter definitions
├── data/
│   ├── raw_articles/
│   │   └── articles/             # Raw source documents (e.g., PDFs)
│   └── vector_db/                # Local ChromaDB persistence directory
├── demo/
│   ├── app.py                    # Streamlit visual interface
│   └── llm.py                    # LLM instatiation script
└── src/
    ├── core/
    │   ├── entities.py           # Domain entities (e.g., Document dataclass)
    │   └── interfaces.py         # Stable contracts (BaseRetriever, BaseDocumentLoader, etc.)
    ├── domain/
    │   ├── ingestion_manager.py  # IngestionEngine (orchestrates loading, splitting, hashing, and saving)
    │   └── rag_engine.py         # RAGEngine (orchestrates query processing, retrieval, and ranking)
    └── infrastructure/
        ├── langchain_impl/       # Adapters for LangChain (loaders, splitters, embeddings, processors)
        └── vector_store/         # Concrete vector database implementations (ChromaDB)

```

## Quick Start

### 1. Setup Environment

Since this is a monorepo, all dependencies are managed at the root.

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies from the monorepo root
pip install -r requirements.txt

```

### 2. Run the Visual Demo

You can test the ingestion and retrieval pipeline interactively using the Streamlit demo:

```bash
# Execute from the monorepo root
streamlit run knowledge_module/demo/app.py

```

## Configuration

Hyperparameters, model choices, and structural settings are externalized to avoid hardcoded values.

Edit `config/models.yaml`:

```yaml
models:
  embedding:
    model: "all-MiniLM-L6-v2"
    provider: "huggingface"
    key: "INSERT API KEY"
  splitter:
    chunk_size: 500
    chunk_overlap: 100

```

## Usage

The module exposes two main domain engines: `IngestionEngine` and `RAGEngine`.

### Document Ingestion

```python
from pathlib import Path
from src.domain.ingestion_manager import IngestionEngine
from src.infrastructure.langchain_impl.loaders import LCLoader
from src.infrastructure.langchain_impl.splitters import LCSplitter
from src.infrastructure.vector_store.chroma_db import LCChromaVectorStore

# 1. Instantiate concrete implementations
loader = LCLoader()
splitter = LCSplitter(chunk_size=500, chunk_overlap=100)
vector_store = LCChromaVectorStore(persist_path="./data/vector_db", embedding_model=my_embedder)

# 2. Inject dependencies into the domain engine
engine = IngestionEngine(loader=loader, splitter=splitter, saver=vector_store)

# 3. Ingest data (automatically handles hashing and prevents duplication)
engine.ingest_content(Path("data/raw_articles/sample"))

```

### Context Retrieval

```python
from src.domain.rag_engine import RAGEngine
from src.infrastructure.langchain_impl.processor import PassthroughQueryProcessor, PassthroughPostProcessor

# 1. Instantiate processors and retriever
query_processor = PassthroughQueryProcessor()
post_processor = PassthroughPostProcessor()
# vector_store acts as the retriever here

# 2. Initialize the RAG engine
rag = RAGEngine(query_processor, vector_store, post_processor)

# 3. Retrieve context for the Reasoning Module
query = "What conclusions do the authors draw regarding the QM9 dataset?"
retrieved_docs = rag.retrieve_context(query)

for doc in retrieved_docs:
    print(f"Content: {doc.page_content}")
    print(f"Source: {doc.metadata['source']}")

```

## Output Contract

The core entity passing through the system is the `Document` dataclass. This acts as the stable output contract to feed the Reasoning Module.

```python
@dataclass
class Document:
    page_content: str   # Text retrieved
    metadata: dict      # Article metadata (source, page, author, etc.)
    score: float = 0.0  # Similarity score

```

## Testing Strategy

To ensure stability across the monorepo, tests rely on validating the architectural contracts rather than testing third-party tools directly.

* **Contracts Verification**: Ensuring classes properly implement interfaces like `BaseDocumentLoader` and `BaseDocumentSaver`.
* **Mocking**: The `IngestionEngine` and `RAGEngine` are tested by injecting `Mock` objects (e.g., `Mock(spec=BaseDocumentSaver)`) to ensure the correct data flows through the pipeline without relying on a live vector database or real embedding models.

## Best Practices

* All code, documentation, variables, and commit messages must be strictly in English.
* Do not instantiate LangChain objects directly inside the domain engines (`rag_engine.py`, `ingestion_manager.py`); strictly use Dependency Injection via the `src.core.interfaces`.
* When writing tests for the pipeline, use the `hash_ids` enforcement to prevent regression in the chunk duplication protection logic.
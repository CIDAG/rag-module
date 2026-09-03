from pathlib import Path
from src.domain.ingestion_manager import IngestionEngine
from src.domain.rag_engine import RAGEngine
from src.infrastructure.langchain_impl.embeddings import LCEmbedding
from src.infrastructure.langchain_impl.loaders import LCLoader
from src.infrastructure.langchain_impl.processor import PassthroughPostProcessor, PassthroughQueryProcessor
from src.infrastructure.langchain_impl.splitters import LCSplitter
from src.infrastructure.vector_store.chroma_db import LCChromaVectorStore

def test_retrieval_pipeline(
        rag_engine: RAGEngine, 
        ingestion_engine: IngestionEngine,
        docs_path: Path | str
) -> None:
    # Ingestion phase
    print(f"Ingesting docs at: {docs_path}")
    ingestion_engine.ingest_content(docs_path)
    print(f"Ingestion completed!")
    
    # Retrieval phase
    query = "Upon analyzing the QM9 dataset, what conclusions do the authors draw regarding the lack of direct correlation between polarizability and the HOMO-LUMO energy gap in the context of rational drug design?"
    print(f"Retrieving articles related to the query: '{query}'")
    retrieved_texts = rag_engine.retrieve_context(query)
    print(f"{len(retrieved_texts)} were retrieved:")
    for i, doc in enumerate(retrieved_texts):
        print(f"\n--- Text {i} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

def get_engines(persist_path: Path | str) -> tuple[RAGEngine, IngestionEngine]:
    from config.config import get_embedding_config, get_splitting_config
    from src.infrastructure.langchain_impl.embeddings import LCEmbedding
    from src.infrastructure.langchain_impl.loaders import LCLoader
    from src.infrastructure.langchain_impl.processor import PassthroughPostProcessor, PassthroughQueryProcessor
    from src.infrastructure.langchain_impl.splitters import LCSplitter
    from src.infrastructure.vector_store.chroma_db import LCChromaVectorStore
    SPLITTER_CONFIGS = get_splitting_config()
    EMBEDDING_CONFIGS = get_embedding_config()

    # Specific implementations
    embedding = LCEmbedding(
        provider=EMBEDDING_CONFIGS['provider'],
        model_name=EMBEDDING_CONFIGS['model']
    )
    loader = LCLoader()
    processor = PassthroughQueryProcessor()
    post_processor = PassthroughPostProcessor()
    splitter = LCSplitter(
        chunk_size=SPLITTER_CONFIGS['chunk_size'],
        chunk_overlap=SPLITTER_CONFIGS['chunk_overlap']
    )
    vector_store = LCChromaVectorStore(
        persist_path=persist_path,
        embedding_model=embedding
    )

    rag_engine = RAGEngine(
        query_processor=processor,
        retriever=vector_store,
        post_processor=post_processor
    )
    ingestion_engine = IngestionEngine(
        loader=loader,
        splitter=splitter,
        saver=vector_store
    )

    return rag_engine, ingestion_engine

def main():
    from config.config import get_embedding_config, get_splitting_config
    MODULE_PATH = Path(__file__).parent
    PERSIST_PATH = MODULE_PATH / "data" / "vector_db"
    ARTICLES_PATH = MODULE_PATH / "data" / "raw_articles"
    SAMPLE_PATH = ARTICLES_PATH / "sample"
    DOCS_PATH = ARTICLES_PATH / "articles"

    rag_engine, ingestion_engine = get_engines(PERSIST_PATH)
    # Testing phase
    test_retrieval_pipeline(rag_engine, ingestion_engine, SAMPLE_PATH)

if __name__ == "__main__":
    main()
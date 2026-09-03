import sys
import streamlit as st
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

ROOT_PATH = Path(__file__).resolve().parent.parent
PERSIST_PATH = ROOT_PATH / "data" / "vector_db"
SAMPLE_ARTICLES_PATH = ROOT_PATH / "data" / "raw_articles" / "sample"

if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))
from src.domain.rag_engine import RAGEngine
from src.domain.ingestion_manager import IngestionEngine
from demo.llm import init_llm

@st.cache_resource
def get_engines(persist_path: Path | str=PERSIST_PATH) -> tuple[RAGEngine, IngestionEngine]:
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

@st.cache_resource
def ingest(_ingestion_engine):
    _ingestion_engine.ingest_content(SAMPLE_ARTICLES_PATH)

def get_chain(retrieved_context: str):
    system_message = '''
    Você é um assistente de pesquisa.
    Você possui acesso às seguintes informações vindas 
    de um documento: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário que houve um erro no carregamento do modelo.
    '''.format(retrieved_context)

    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    model = init_llm()
    chain = template | model
    return chain

st.title("📚 Demonstração visual módulo RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça uma pergunta sobre seus documentos:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        rag, ingestion = get_engines()
        ingest(ingestion)
        docs_retrieved = rag.retrieve_context(prompt)

        context_str = "\n\n".join([f"Fragmento:\n{doc.page_content}" for doc in docs_retrieved])
        st.caption(f"Contexto recuperado de {len(docs_retrieved)} fragmento(s).")
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            role = "user" if msg["role"] == "user" else "assistant"
            chat_history.append((role, msg["content"]))

        chain = get_chain(context_str)
        response_stream = chain.stream({
            "context": context_str,
            "chat_history": chat_history,
            "input": prompt
        }) 
        response = st.write_stream(response_stream)
        fragments = f"Fragmentos de texto:\n\n{context_str}"
        fragments_printed = st.markdown(fragments)
        
        st.session_state.messages.append({"role": "assistant", "content": f"{response}\n{fragments}"})

# PERGUNTAS:
# Upon evaluating the citation patterns of papers published in the Journal of Chemical Information and Modeling (JCIM) from 1996 to 2022, what conclusions do the authors draw regarding the homophilic citation characteristic among female researchers in the context of achieving future gender equity in scientific publications?
# 
# Upon assessing the performance of the Supervised Grammar Variational Autoencoder (SGVAE) on the QM9 dataset, what explanations do the authors provide regarding the significant disparity in the generation of novel and unique molecules when using SMILES from DFT-B3LYP relaxed geometries versus those from the GDB-17 database?
#
# Upon investigating various data augmentation techniques within the SMICLR framework, what justifications do the authors provide regarding the severe deterioration in predictive performance caused by subgraph extraction in the context of the average molecule size in the QM9 dataset?
#
# Upon exploring the learned representations of the deep tensor neural network (DTNN), what observations do the authors make regarding the model's ability to grasp fundamental chemical concepts, such as the high aromatic ring stability of $C_6O_3H_6$, in the context of models trained exclusively on molecular total energies? 
#
# Upon analyzing the geographic and disciplinary distribution of JCIM manuscripts, what trends do the authors identify regarding the impact of international collaboration and multi-area research on average annual citation rates?
#
# Upon analyzing the QM9 dataset, what conclusions do the authors draw regarding the lack of direct correlation between polarizability and the HOMO-LUMO energy gap in the context of rational drug design?

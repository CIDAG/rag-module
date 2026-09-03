# Arquitetura do Módulo 3 (Knowledge / RAG) – Draft

Este documento descreve a arquitetura interna, os contratos e a estrutura de pastas do **Módulo 3 (Knowledge)**, responsável por fornecer contexto científico recuperado para o Módulo 2 (Reasoning) do Chart Analysis Framework.

## 1. Visão geral do Módulo

O Módulo de Conhecimento atua como o motor de **Retrieval-Augmented Generation (RAG)** do framework. Seu objetivo é indexar artigos científicos (textos, tabelas e contextos teóricos) e fornecer trechos relevantes e metadados que auxiliem o LLM (Módulo 2) a interpretar com maior precisão os dados numéricos extraídos dos gráficos pelo Módulo 1.

Para garantir escalabilidade e permitir a experimentação acadêmica com diferentes técnicas de RAG (do Ingênuo ao Avançado), o módulo foi desenhado utilizando princípios de **Orientação a Objetos** e **Clean Architecture** (Arquitetura Limpa). A dependência de bibliotecas externas, como o LangChain e bancos de dados vetoriais, é estritamente isolada da regra de negócio central.

## 2. Estrutura de pastas atual (Módulo 3)

Abaixo, a estrutura da subpasta `knowledge_module/` dentro do monorepo.

```text
knowledge_module/                 
├── config/
│   ├── config.py               # Carregador das configurações
│   └── models.yaml             # Configurações globais
├── data/
│   ├── raw_articles/           # Arquivos físicos brutos (PDFs, JSONs, LaTeX) - Ignorado no Git
│   └── vector_db/              # Persistência do banco vetorial local (ex: ChromaDB) - Ignorado no Git
├── src/
│   ├── core/                   # CONTRATOS: Regras de negócio sem dependências externas
│   │   ├── interfaces.py       # Classes abstratas (BaseQueryProcessor, BaseRetriever, BasePostProcessor)
│   │   └── entities.py         # Modelos de dados puros (ex: classe Document)
│   ├── infrastructure/         # IMPLEMENTAÇÃO: Adaptadores para LangChain, bancos vetoriais, etc.
│   │   ├── langchain_impl/     # Wrappers para loaders, splitters e embeddings do LangChain
│   │   └── vector_store/       # Gerenciamento de conexão com o banco (ex: ChromaManager)
│   └── domain/                 # ORQUESTRAÇÃO: Lógica do pipeline RAG
│       ├── ingestion_engine.py # Orquestra o fluxo de ETL (Leitura -> Chunking -> Embedding -> DB)
│       └── rag_engine.py       # Orquestra a busca em tempo real (Query -> Retriever -> Reranker)
├── run.py                      # Ponto de entrada (orquestração e comunicação com o Orchestrator)
├── .gitignore
├── requirements.txt            # Dependências isoladas do módulo (langchain, chromadb, etc.)
└── README.md                   # Este documento
```

## 3. Papel de cada componente

| Componente | Função |
|------------|--------|
| **config/** | Centraliza variáveis de ambiente, caminhos de diretórios (usando `pathlib`) e hiperparâmetros de chunking e busca. |
| **data/** | Armazena tanto o acervo de literatura científica original (`raw_articles/`) quanto os dados processados e indexados (`vector_db/`). |
| **src/core/** | Define as interfaces (`ABC`) do sistema. Garante que qualquer modelo ou banco de dados obedeça a um contrato padrão, permitindo testes ágeis sem quebrar o módulo. |
| **src/infrastructure/** | Contém o código que efetivamente interage com bibliotecas de terceiros (LangChain). Se uma tecnologia mudar, apenas esta pasta sofre manutenção. |
| **src/domain/** | Contém a máquina lógica (`RAGEngine`). Recebe as implementações da infraestrutura via Injeção de Dependência e executa o pipeline do RAG. |
| **run.py** | Atua como a Fachada (Facade) do módulo. É o arquivo que o `orchestrator/` do monorepo irá importar para consumir o serviço de busca. |

## 4. Fluxo interno do Módulo 3

O processamento é dividido em dois ciclos independentes:

**Ciclo de Ingestão (Offline):**
1. O `ingestion_engine` lê os arquivos da pasta `data/raw_articles/`.
2. Utiliza os *loaders* e *splitters* da infraestrutura para quebrar o texto.
3. Salva os embeddings no banco vetorial (`data/vector_db/`).

**Ciclo de Recuperação (Tempo Real):**
1. O `run.py` recebe a *query* (pergunta do usuário e/ou dados do gráfico).
2. O `RAGEngine` processa/expande a *query*.
3. O *Retriever* busca no banco de dados e retorna uma lista de `Document`s brutos.
4. O *PostProcessor* filtra/reordena os documentos (Reranking).
5. O `run.py` devolve os documentos formatados para o Orchestrator encaminhar ao Módulo 2.
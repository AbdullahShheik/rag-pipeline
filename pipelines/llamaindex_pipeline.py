import os

from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from sqlalchemy.engine.url import make_url

PDF_PATH = "/docs/CoMuRoS_LLM-Based_Generalizable_Hierarchical_Task_Planning_and_Execution_for_Heterogeneous_Robot_Teams_with_.pdf"
EMBEDDING_DIM = 3072
TABLE_NAME = "llamaindex_chunks"

SPLITTER = SentenceSplitter(chunk_size=500, chunk_overlap=50)


def get_embed_model():
    return GoogleGenAIEmbedding(
        model_name="gemini-embedding-2-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_llm():
    return GoogleGenAI(
        model="gemini-3.6-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _pg_conn_kwargs():
    url = make_url(os.getenv("DATABASE_URL"))
    return dict(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port or 5432,
        user=url.username,
    )


def get_vector_store():
    return PGVectorStore.from_params(
        **_pg_conn_kwargs(),
        table_name=TABLE_NAME,
        embed_dim=EMBEDDING_DIM,
    )


def get_docstore():
    return PostgresDocumentStore.from_params(
        **_pg_conn_kwargs(),
        namespace="comuros",
    )


def get_index_store():
    return PostgresIndexStore.from_params(
        **_pg_conn_kwargs(),
        namespace="comuros",
    )


def get_storage_context():
    return StorageContext.from_defaults(
        docstore=get_docstore(),
        index_store=get_index_store(),
        vector_store=get_vector_store(),
    )


def load_documents(pdf_path: str):
    reader = PDFReader()
    docs = reader.load_data(file=pdf_path)
    for i, doc in enumerate(docs):
        doc.doc_id = f"{os.path.basename(pdf_path)}::page-{i}"
    return docs


def get_or_build_index(storage_context, embed_model):
    documents = load_documents(PDF_PATH)

    try:
        index = load_index_from_storage(storage_context)
        print("Loaded existing index from storage.")

        refreshed = index.refresh_ref_docs(
            documents,
            update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
        )
        n_changed = sum(refreshed)
        print(f"refresh_ref_docs: {n_changed} of {len(documents)} documents were new/changed and re-embedded.")

    except ValueError:
        print(f"No existing index found. Building fresh index from {PDF_PATH} ...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SPLITTER],
            embed_model=embed_model,
        )
        print(f"Indexed {len(documents)} documents.")

    return index


if __name__ == "__main__":
    embed_model = get_embed_model()
    llm = get_llm()
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.node_parser = SPLITTER  

    storage_context = get_storage_context()
    index = get_or_build_index(storage_context, embed_model)

    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

    question = "What is CoMuRoS and what problem does it solve?"
    response = query_engine.query(question)

    print(f"\nRetrieved {len(response.source_nodes)} chunks for question: '{question}'\n")
    print("---- Generated Answer ----")
    print(str(response))
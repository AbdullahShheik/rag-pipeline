import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from langchain_google_genai import ChatGoogleGenerativeAI

PDF_PATH = "/docs/CoMuRoS_LLM-Based_Generalizable_Hierarchical_Task_Planning_and_Execution_for_Heterogeneous_Robot_Teams_with_.pdf"
EMBEDDING_DIM = 3072


Base = declarative_base()


class LangchainChunk(Base):
    __tablename__ = "langchain_chunks"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))


engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(bind=engine)


def get_session():
    return SessionLocal()


def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def store_chunks(chunks, embedder):
    session = get_session()

    try:
        for chunk in chunks:
            vector = embedder.embed_query(chunk.page_content)
            db_chunk = LangchainChunk(content=chunk.page_content, embedding=vector)
            session.add(db_chunk)

        session.commit()
        print(f"Stored {len(chunks)} chunks in pgvector.")
    except Exception as e:
        session.rollback()
        print(f"Error storing chunks, rolled back: {e}")
        raise
    finally:
        session.close()

def retrieve_top_k(question: str, embedder, k: int = 3):
    question_vector = embedder.embed_query(question)

    session = get_session()
    try:
        results = (
            session.query(LangchainChunk)
            .order_by(LangchainChunk.embedding.cosine_distance(question_vector))
            .limit(k)
            .all()
        )
        return results
    finally:
        session.close()


def generate_answer(question: str, retrieved_chunks, llm):
    context = "\n\n".join(chunk.content for chunk in retrieved_chunks)

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    
    if isinstance(response.content, list):
        return "".join(
            block["text"] for block in response.content 
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return response.content

if __name__ == "__main__":
    Base.metadata.create_all(engine)

    embedder = get_embedder()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    session = get_session()
    try:
        already_ingested = session.query(LangchainChunk).first() is not None
    finally:
        session.close()

    if not already_ingested:
        print(f"No chunks found. Ingesting from {PDF_PATH} ...")
        chunks = load_and_chunk(PDF_PATH)
        store_chunks(chunks, embedder)
    else:
        print("Chunks already present in DB, skipping ingestion.")

    question = "What is CoMuRoS and what problem does it solve?"
    top_chunks = retrieve_top_k(question, embedder, k=3)

    print(f"\nRetrieved {len(top_chunks)} chunks for question: '{question}'\n")

    answer = generate_answer(question, top_chunks, llm)
    print("---- Generated Answer ----")
    print(answer)
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

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


if __name__ == "__main__":
    Base.metadata.create_all(engine)

    chunks = load_and_chunk(PDF_PATH)
    print(f"Loaded {len(chunks)} chunks from PDF")

    embedder = get_embedder()

    store_chunks(chunks, embedder)

# ingestion_pipeline.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from embeddings_ollama import OllamaEmbeddings

load_dotenv()


class UTF8TextLoader(TextLoader):
    """TextLoader variant that always uses UTF-8 encoding on Windows."""
    def __init__(self, file_path, encoding="utf-8", autodetect_encoding=False):
        super().__init__(file_path, encoding=encoding, autodetect_encoding=autodetect_encoding)


def load_job_documents(jobs_path="offres"):
    print(f"Loading job offers from {jobs_path}...")
    if not os.path.exists(jobs_path):
        raise FileNotFoundError(f"Directory not found: {jobs_path}")

    loader = DirectoryLoader(
        path=jobs_path,
        glob="*.txt",
        loader_cls=UTF8TextLoader
    )
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {jobs_path}")

    # Tag metadata
    for d in docs:
        d.metadata["doc_type"] = "job"

    print(f"Loaded {len(docs)} job documents.")
    return docs


def load_cv_documents(cvs_path="CV"):
    print(f"Loading CVs from {cvs_path}...")
    if not os.path.exists(cvs_path):
        raise FileNotFoundError(f"Directory not found: {cvs_path}")

    docs = []
    for fn in os.listdir(cvs_path):
        if fn.lower().endswith(".pdf"):
            fp = os.path.join(cvs_path, fn)
            loader = PyPDFLoader(fp)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["doc_type"] = "cv"
                d.metadata["cv_file"] = fn
            docs.extend(pdf_docs)

    if not docs:
        raise FileNotFoundError(f"No .pdf files found in {cvs_path}")

    print(f"Loaded {len(docs)} CV pages (PDF pages).")
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    print("Splitting documents into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")
    return chunks


def build_or_load_vectorstore(persist_directory, collection_name, documents=None):
    """
    If persist_directory exists AND has data, loads it.
    Otherwise builds from documents and persists.
    """
    embedding_model = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "local-embedding")
    )

    os.makedirs(persist_directory, exist_ok=True)

    # Load existing collection if already indexed
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = db._collection.count()
    if count > 0:
        print(f"✅ Loaded existing vectorstore '{collection_name}' with {count} chunks.")
        return db

    if documents is None:
        raise ValueError(f"No existing DB found for '{collection_name}' and no documents provided to build it.")

    print(f"Creating vectorstore '{collection_name}' in {persist_directory} ...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"✅ Built vectorstore '{collection_name}' with {db._collection.count()} chunks.")
    return db


def main():
    print("=== Smart Resume & Job Matcher | Ingestion ===\n")

    jobs_path = "offres"
    cvs_path = "CV"

    # Two separate DB folders (clean separation)
    jobs_db_dir = os.path.join("db", "chroma_jobs")
    cvs_db_dir = os.path.join("db", "chroma_cvs")

    # 1) Jobs
    job_docs = load_job_documents(jobs_path)
    job_chunks = split_documents(job_docs, chunk_size=1000, chunk_overlap=150)
    jobs_db = build_or_load_vectorstore(
        persist_directory=jobs_db_dir,
        collection_name="jobs",
        documents=job_chunks
    )

    # 2) CVs
    cv_docs = load_cv_documents(cvs_path)
    cv_chunks = split_documents(cv_docs, chunk_size=900, chunk_overlap=150)
    cvs_db = build_or_load_vectorstore(
        persist_directory=cvs_db_dir,
        collection_name="cvs",
        documents=cv_chunks
    )

    print("\n✅ Ingestion complete.")
    print(f"- Jobs chunks: {jobs_db._collection.count()}  (db: {jobs_db_dir})")
    print(f"- CV chunks:   {cvs_db._collection.count()}  (db: {cvs_db_dir})")
    return jobs_db, cvs_db


if __name__ == "__main__":
    main()

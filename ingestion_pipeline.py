import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import requests


class OllamaEmbeddings:
    """Minimal embeddings wrapper for Ollama's local HTTP API.

    This implements `embed_documents` and `embed_query` methods expected
    by LangChain/Chroma. It posts to `http://localhost:11434/api/embeddings`.
    Configure model with the `OLLAMA_EMBEDDING_MODEL` env var.
    """

    def __init__(self, model=None, base_url=None, timeout=30):
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "local-embedding")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

    def _call_ollama(self, inputs):
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "input": inputs}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to call Ollama embeddings API at {url}: {e}")

        data = resp.json()

        # Normalize responses of common shapes
        embeddings = []
        if isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                emb = item.get("embedding") or item.get("embeddings")
                if emb is None:
                    raise RuntimeError(f"No embedding found in response item: {item}")
                embeddings.append(emb)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                else:
                    raise RuntimeError(f"Unexpected embedding list item: {item}")
        else:
            raise RuntimeError(f"Unexpected response shape from Ollama embeddings API: {data}")

        return embeddings

    def embed_documents(self, texts):
        return self._call_ollama(texts)

    def embed_query(self, text):
        return self._call_ollama([text])[0]

from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class UTF8TextLoader(TextLoader):
    """TextLoader variant that always uses UTF-8 encoding on Windows."""
    def __init__(self, file_path, encoding="utf-8", autodetect_encoding=False):
        super().__init__(file_path, encoding=encoding, autodetect_encoding=autodetect_encoding)


def load_documents(docs_path="offres"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory with UTF-8 encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=UTF8TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "local-embedding"))
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "LLM_Smart_Resume_Job_Master\offres"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "local-embedding"))
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()

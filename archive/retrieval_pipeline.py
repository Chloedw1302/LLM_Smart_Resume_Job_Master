import os
import requests
from langchain_chroma import Chroma
from dotenv import load_dotenv


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


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "local-embedding"))

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
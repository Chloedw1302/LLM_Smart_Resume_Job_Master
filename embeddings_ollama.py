# embeddings_ollama.py
import os
import requests


class OllamaEmbeddings:
    """
    Ollama embeddings wrapper (compatible with your server):
    Uses: POST /api/embeddings with {"model": "...", "prompt": "..."}
    Returns: {"embedding": [...]}
    """

    def __init__(self, model=None, base_url=None, timeout=60):
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

    def _embed_one(self, text: str):
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Unexpected response from Ollama: {data}")
        return data["embedding"]

    def embed_documents(self, texts):
        # one call per chunk = reliable across versions
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text):
        return self._embed_one(text)

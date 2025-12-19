import os
import requests

def ollama_generate(prompt: str, timeout: int = 60) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url}/api/generate"

    payload = {
        "model": "phi3:mini",     # force fast model
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 512,
            "num_predict": 170,   # assez pour 5 puces
            # stop dès qu'il commence à sortir du format
            "stop": ["\n\n", "Résumé", "Objectif", "Stage", "Traduction", "Conclusion"]
        }
    }

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()
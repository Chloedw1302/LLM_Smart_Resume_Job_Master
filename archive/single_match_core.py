import math
from typing import Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from embeddings_ollama import OllamaEmbeddings
from llm_structuring import extract_cv_json, extract_job_json
from llm_explain import explain_match


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def match_percent_from_cosine(c: float) -> float:
    # cosine usually in [0,1] here; clip and scale
    c = max(0.0, min(1.0, c))
    return 100.0 * c


def verdict(pct: float) -> str:
    if pct >= 70:
        return "✅ Postuler"
    if pct >= 50:
        return "⚠️ À tenter (optimise ton CV)"
    return "❌ Pas prioritaire"


def best_snippet(text: str, keywords, fallback_len: int = 320) -> str:
    low = text.lower()
    for kw in keywords:
        kw = str(kw).lower().strip()
        if not kw:
            continue
        i = low.find(kw)
        if i != -1:
            start = max(0, i - 140)
            end = min(len(text), i + 220)
            return " ".join(text[start:end].split())
    return " ".join(text[:fallback_len].split())


def load_cv_pdf_text(pdf_path: str, max_chars: int = 9000) -> str:
    docs = PyPDFLoader(pdf_path).load()
    text = "\n".join(d.page_content for d in docs)
    return text[:max_chars]


def single_match(cv_pdf_path: str, offer_text: str) -> Dict[str, Any]:
    # 1) Load CV
    cv_text = load_cv_pdf_text(cv_pdf_path)

    # 2) Embeddings similarity
    emb = OllamaEmbeddings()
    cv_vec = emb.embed_query(cv_text[:3500])
    offer_vec = emb.embed_query(offer_text[:3500])
    cos = cosine(cv_vec, offer_vec)
    pct = match_percent_from_cosine(cos)

    # 3) LLM structuring (small excerpts -> fast)
    cv_json = extract_cv_json(cv_text[:2500])
    job_json = extract_job_json(offer_text[:2500])

    # 4) Evidence snippets
    kw = (cv_json.get("skills", [])[:12] + job_json.get("required_skills", [])[:12])
    cv_snip = best_snippet(cv_text, kw)
    offer_snip = best_snippet(offer_text, kw)

    # 5) LLM explanation (3 match + gap + advice)
    explanation = explain_match(cv_json, job_json, cv_snip, offer_snip)

    return {
        "cosine": float(cos),
        "match_percent": float(pct),
        "verdict": verdict(pct),
        "cv_json": cv_json,
        "job_json": job_json,
        "cv_snippet": cv_snip,
        "offer_snippet": offer_snip,
        "llm_explanation": explanation,
    }
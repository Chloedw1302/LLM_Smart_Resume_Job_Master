import math
from langchain_community.document_loaders import PyPDFLoader
from embeddings_ollama import OllamaEmbeddings
from requests.exceptions import ReadTimeout
from ollama_chat import ollama_generate
from explain_rules import explain_match_rules


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def match_percent_from_cosine(c):
    c = max(0.0, min(1.0, c))
    return 100.0 * c

def pick_cv_excerpt(text: str, max_len=900) -> str:
    low = text.lower()
    for marker in ["experience", "expérience", "compétences", "skills", "projets", "projects"]:
        i = low.find(marker)
        if i != -1:
            return " ".join(text[i:i+max_len].split())
    return " ".join(text[:max_len].split())

def verdict(pct):
    if pct >= 70:
        return "✅ Postuler"
    if pct >= 50:
        return "⚠️ À tenter (adapter le CV)"
    return "❌ Pas prioritaire"


def load_cv_pdf_text(pdf_path, max_chars=6000):
    docs = PyPDFLoader(pdf_path).load()
    text = "\n".join(d.page_content for d in docs)
    return text[:max_chars]


def single_match_fast(cv_pdf_path: str, offer_text: str):
    # 1) Load texts
    cv_text = load_cv_pdf_text(cv_pdf_path)
    cv_excerpt = pick_cv_excerpt(cv_text, 900)
    offer_text = offer_text[:6000]

    # 2) Embeddings similarity (FAST)
    emb = OllamaEmbeddings()
    cv_vec = emb.embed_query(cv_text[:3000])
    offer_vec = emb.embed_query(offer_text[:3000])
    cos = cosine(cv_vec, offer_vec)

    pct = match_percent_from_cosine(cos)
    decision = verdict(pct)

    # 3) ONE LLM CALL: explanation + gap + advice
    prompt = f"""
Réponds UNIQUEMENT avec exactement 5 lignes.
Chaque ligne doit commencer par le préfixe indiqué.

Format OBLIGATOIRE (copie exactement) :
- ✅ Match: ...
- ✅ Match: ...
- ✅ Match: ...
- ⚠️ Gap: ...
- 🎯 Conseil: ...

Règles:
- Pas de titre.
- Pas de résumé.
- Pas de traduction.
- Ne parle que du contenu fourni.
- Si tu ne vois pas l'info, écris "non mentionné".

CV:
\"\"\"{cv_excerpt[:900]}\"\"\"

OFFRE:
\"\"\"{offer_text[:900]}\"\"\"
"""

    explanation = explain_match_rules(cv_text, offer_text, pct)

    return {
        "cosine": cos,
        "match_percent": pct,
        "verdict": decision,
        "llm_explanation": explanation,
    }
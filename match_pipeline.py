# match_pipeline.py (v3)
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from embeddings_ollama import OllamaEmbeddings

load_dotenv()

JOBS_DB_DIR = os.path.join("db", "chroma_jobs")
CVS_DB_DIR = os.path.join("db", "chroma_cvs")

# -----------------------------
# Skills & keywords (enrichis)
# -----------------------------
SKILLS_CANON = [
    # Core data
    "python", "sql", "excel", "power bi", "powerbi", "dax", "tableau",
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "machine learning", "deep learning", "nlp", "llm", "rag",
    "data analysis", "data analytics", "data science", "statistics", "statistiques",
    "feature engineering", "data cleaning", "data quality", "data visualization", "dashboard",
    "etl", "pipeline", "automation", "reporting", "kpi",
    # Tools / eng
    "git", "docker", "linux",
    "postgresql", "mysql", "mongodb",
    "spark", "databricks", "airflow", "dbt",
    "fastapi", "flask", "streamlit",
    "azure", "aws", "gcp", "google cloud",
    # Business systems
    "erp", "sap",
    # Finance
    "valuation", "audit", "accounting", "financial modeling", "m&a", "due diligence",
    # Meca
    "ansys", "comsol", "catia", "solidworks", "fea", "finite element",
]

MISSIONS_KEYWORDS = [
    "dashboard", "reporting", "kpi", "visualization", "data visualization",
    "data cleaning", "data quality", "automation", "etl", "pipeline",
    "analysis", "analytics", "model", "predict", "classification", "regression", "forecast",
    "stakeholder", "business", "process", "optimisation", "optimization",
]

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_keywords(text: str, vocabulary: List[str]) -> List[str]:
    t = normalize_text(text)
    found = []
    for term in vocabulary:
        if term in t:
            found.append(term)
    return sorted(set(found))

def pick_best_snippet(text: str, keywords: List[str], fallback_len: int = 320) -> str:
    if not text:
        return ""
    low = text.lower()
    for kw in keywords:
        idx = low.find(kw.lower())
        if idx != -1:
            start = max(0, idx - 140)
            end = min(len(text), idx + 220)
            return re.sub(r"\s+", " ", text[start:end]).strip()
    return re.sub(r"\s+", " ", text[:fallback_len]).strip()

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

# -----------------------------
# DB load
# -----------------------------
def load_db(persist_directory: str, collection_name: str) -> Chroma:
    emb = OllamaEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=emb,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

# -----------------------------
# CV query
# -----------------------------
def get_cv_text(cvs_db: Chroma, cv_filename: str, max_chars: int = 8000) -> str:
    data = cvs_db.get(where={"cv_file": cv_filename})
    docs = data.get("documents", [])
    if not docs:
        raise ValueError(f"No chunks found for {cv_filename}. Did you ingest CVs?")
    return "\n".join(docs)[:max_chars]

def build_cv_query(cv_text: str, max_chars: int = 3500) -> str:
    cv_text = cv_text[:max_chars]
    return (
        "Candidate profile for matching to internship/job descriptions.\n"
        "Focus on skills, tools, experience, education, projects.\n\n"
        f"{cv_text}"
    )

# -----------------------------
# Offer category filter
# -----------------------------
def infer_cv_category(cv_text: str) -> str:
    t = normalize_text(cv_text)
    data_hits = sum(k in t for k in ["python", "sql", "power bi", "powerbi", "pandas", "data", "dashboard"])
    fin_hits = sum(k in t for k in ["finance", "audit", "valuation", "accounting", "m&a", "due diligence"])
    meca_hits = sum(k in t for k in ["ansys", "comsol", "catia", "solidworks", "mechanical", "vibro", "matériaux"])

    if data_hits >= max(fin_hits, meca_hits) and data_hits > 0:
        return "data"
    if fin_hits >= max(data_hits, meca_hits) and fin_hits > 0:
        return "fin"
    if meca_hits >= max(data_hits, fin_hits) and meca_hits > 0:
        return "meca"
    return "all"

def category_from_source_path(source: str) -> str:
    base = os.path.basename(source).lower()
    if "offre_stage_data" in base:
        return "data"
    if "offre_stage_fin" in base:
        return "fin"
    if "offre_stage_meca" in base:
        return "meca"
    return "other"

# -----------------------------
# Scoring aggregation
# -----------------------------
@dataclass
class OfferMatch:
    source: str
    category: str
    offer_score: float
    match_percent: float
    bullets: List[str]
    cv_snippet: str
    offer_snippet: str

def aggregate_scores_by_source(raw_results, category_filter: Optional[str]) -> Dict[str, List[float]]:
    grouped = defaultdict(list)
    for doc, score in raw_results:
        src = doc.metadata.get("source", "unknown_source")
        cat = category_from_source_path(src)
        if category_filter and category_filter != "all":
            if cat != category_filter:
                continue
        grouped[src].append(float(score))
    return grouped

def score_offer(grouped_scores: Dict[str, List[float]], top_n: int = 3) -> Dict[str, float]:
    offer_scores = {}
    for src, scores in grouped_scores.items():
        best = sorted(scores, reverse=True)[:top_n]
        offer_scores[src] = sum(best) / len(best)
    return offer_scores

def normalize_scores_to_percent(offer_scores: Dict[str, float]) -> Dict[str, float]:
    if not offer_scores:
        return {}
    vals = list(offer_scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {k: 100.0 for k in offer_scores}
    return {k: 100.0 * (v - mn) / (mx - mn) for k, v in offer_scores.items()}

def overlap_percent(cv_skills: List[str], offer_skills: List[str]) -> float:
    if not offer_skills:
        return 0.0
    inter = set(cv_skills) & set(offer_skills)
    return 100.0 * safe_div(len(inter), len(set(offer_skills)))

def final_percent(semantic_percent: float, overlap: float, alpha_semantic: float = 0.75) -> float:
    return alpha_semantic * semantic_percent + (1 - alpha_semantic) * overlap

# -----------------------------
# Get full offer text by joining chunks from same source
# -----------------------------
def get_offer_full_text(jobs_db: Chroma, source: str, max_chars: int = 8000) -> str:
    data = jobs_db.get(where={"source": source})
    docs = data.get("documents", [])
    if not docs:
        return ""
    joined = "\n".join(docs)
    return joined[:max_chars]

# -----------------------------
# Main matching
# -----------------------------
def match_cv_to_offers(cv_filename: str, mode: str = "auto", k_chunks: int = 60, top_offers: int = 5) -> List[OfferMatch]:
    jobs_db = load_db(JOBS_DB_DIR, "jobs")
    cvs_db = load_db(CVS_DB_DIR, "cvs")

    cv_text = get_cv_text(cvs_db, cv_filename)
    cv_query = build_cv_query(cv_text)

    cv_skills = extract_keywords(cv_text, SKILLS_CANON)
    cv_tasks = extract_keywords(cv_text, MISSIONS_KEYWORDS)

    inferred = infer_cv_category(cv_text)
    category_filter = inferred if mode == "auto" else mode

    raw = jobs_db.similarity_search_with_score(cv_query, k=k_chunks)

    grouped_scores = aggregate_scores_by_source(raw, category_filter=category_filter)
    offer_scores = score_offer(grouped_scores, top_n=3)
    sem_pct = normalize_scores_to_percent(offer_scores)

    matches: List[OfferMatch] = []

    for src, sc in offer_scores.items():
        cat = category_from_source_path(src)
        offer_full = get_offer_full_text(jobs_db, src)

        offer_skills = extract_keywords(offer_full, SKILLS_CANON)
        offer_tasks = extract_keywords(offer_full, MISSIONS_KEYWORDS)

        ov = overlap_percent(cv_skills, offer_skills)
        pct = final_percent(sem_pct.get(src, 0.0), ov, alpha_semantic=0.75)

        common_skills = sorted(set(cv_skills) & set(offer_skills))[:8]
        common_tasks = sorted(set(cv_tasks) & set(offer_tasks))[:8]

        bullets = []
        if common_skills:
            bullets.append(f"Skills alignment: {', '.join(common_skills)}.")
        else:
            bullets.append("Skills alignment: strong semantic match even if wording differs.")

        if common_tasks:
            bullets.append(f"Mission alignment: {', '.join(common_tasks)}.")
        else:
            bullets.append("Mission alignment: inferred from similarity between experience and job mission.")

        bullets.append(f"Match% combines semantic similarity + keyword overlap (filtered: {category_filter}).")

        # Evidence snippets
        cv_snip = pick_best_snippet(cv_text, common_skills + common_tasks + cv_skills + cv_tasks)
        offer_snip = pick_best_snippet(offer_full, common_skills + common_tasks + offer_skills + offer_tasks)

        matches.append(
            OfferMatch(
                source=src,
                category=cat,
                offer_score=sc,
                match_percent=pct,
                bullets=bullets,
                cv_snippet=cv_snip,
                offer_snippet=offer_snip,
            )
        )

    matches.sort(key=lambda m: m.match_percent, reverse=True)
    return matches[:top_offers]

def list_cvs() -> List[str]:
    return sorted([f for f in os.listdir("CV") if f.lower().endswith(".pdf")])

def pretty_print(cv_filename: str, matches: List[OfferMatch]):
    print("\n" + "=" * 92)
    print(f"CV: {cv_filename}")
    print("Top matching offers (unique files)")
    print("=" * 92)

    for i, m in enumerate(matches, 1):
        print(f"\n#{i}  {os.path.basename(m.source)}   [{m.category}]")
        print(f"Match: {m.match_percent:.1f}%   (semantic avg score: {m.offer_score:.4f})")
        print("Why this match:")
        for b in m.bullets:
            print(f" - {b}")
        print("\nCV evidence:")
        print(f' "{m.cv_snippet}"')
        print("\nOffer evidence:")
        print(f' "{m.offer_snippet}"')
        print("-" * 92)

if __name__ == "__main__":
    cvs = list_cvs()
    for cv in cvs:
        matches = match_cv_to_offers(cv, mode="auto", k_chunks=80, top_offers=5)
        pretty_print(cv, matches)

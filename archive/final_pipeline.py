import os
import re
from collections import defaultdict
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma

from embeddings_ollama import OllamaEmbeddings
from llm_structuring import extract_cv_json, extract_job_json
from llm_explain import explain_match

from ollama_chat import ollama_generate

# Warmup (helps first request)
_ = ollama_generate("Say 'ready' and nothing else.", timeout=600)


load_dotenv()

JOBS_DB_DIR = os.path.join("db", "chroma_jobs")
CVS_DB_DIR = os.path.join("db", "chroma_cvs")

# ---------- small helpers ----------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pick_snippet(text: str, keywords: List[str], fallback_len: int = 320) -> str:
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

def category_from_source(source: str) -> str:
    base = os.path.basename(source).lower()
    if "offre_stage_data" in base:
        return "data"
    if "offre_stage_fin" in base:
        return "fin"
    if "offre_stage_meca" in base:
        return "meca"
    return "other"

def infer_cv_category(cv_text: str) -> str:
    t = normalize_text(cv_text)
    data_hits = sum(k in t for k in ["python", "sql", "power bi", "pandas", "data", "dashboard"])
    fin_hits = sum(k in t for k in ["finance", "audit", "valuation", "accounting", "m&a", "due diligence"])
    meca_hits = sum(k in t for k in ["ansys", "comsol", "catia", "solidworks", "mechanical", "vibro"])
    if data_hits >= max(fin_hits, meca_hits) and data_hits > 0:
        return "data"
    if fin_hits >= max(data_hits, meca_hits) and fin_hits > 0:
        return "fin"
    if meca_hits >= max(data_hits, fin_hits) and meca_hits > 0:
        return "meca"
    return "all"

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

# ---------- DB ----------
def load_db(persist_directory: str, collection_name: str) -> Chroma:
    emb = OllamaEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=emb,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

def get_cv_text(cvs_db: Chroma, cv_filename: str, max_chars: int = 9000) -> str:
    data = cvs_db.get(where={"cv_file": cv_filename})
    docs = data.get("documents", [])
    if not docs:
        raise ValueError(f"No chunks found for CV: {cv_filename}")
    return "\n".join(docs)[:max_chars]

def build_cv_query(cv_text: str, max_chars: int = 3500) -> str:
    cv_text = cv_text[:max_chars]
    return (
        "Candidate profile for matching to internship/job descriptions.\n"
        "Focus on skills, tools, experience, education, projects.\n\n"
        f"{cv_text}"
    )

def get_offer_full_text(jobs_db: Chroma, source: str, max_chars: int = 9000) -> str:
    data = jobs_db.get(where={"source": source})
    docs = data.get("documents", [])
    return "\n".join(docs)[:max_chars] if docs else ""

# ---------- scoring ----------
def aggregate_scores_by_source(raw_results, category_filter: Optional[str]) -> Dict[str, List[float]]:
    grouped = defaultdict(list)
    for doc, score in raw_results:
        src = doc.metadata.get("source", "unknown_source")
        cat = category_from_source(src)
        if category_filter and category_filter != "all" and cat != category_filter:
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

def overlap_percent(cv_skills: List[str], offer_required: List[str]) -> float:
    if not offer_required:
        return 0.0
    inter = set(map(str.lower, cv_skills)) & set(map(str.lower, offer_required))
    return 100.0 * safe_div(len(inter), len(set(map(str.lower, offer_required))))

def final_percent(semantic_percent: float, overlap: float, alpha_semantic: float = 0.75) -> float:
    return alpha_semantic * semantic_percent + (1 - alpha_semantic) * overlap

# ---------- main pipeline ----------
def run_for_cv(cv_filename: str, mode: str = "auto", k_chunks: int = 80, top_offers: int = 5, explain_top_n: int = 3):
    jobs_db = load_db(JOBS_DB_DIR, "jobs")
    cvs_db = load_db(CVS_DB_DIR, "cvs")

    # 1) CV text
    cv_text = get_cv_text(cvs_db, cv_filename)
    cv_query = build_cv_query(cv_text)

    # 2) Retrieve candidates
    inferred = infer_cv_category(cv_text)
    category_filter = inferred if mode == "auto" else mode

    raw = jobs_db.similarity_search_with_score(cv_query, k=k_chunks)
    grouped_scores = aggregate_scores_by_source(raw, category_filter=category_filter)
    offer_scores = score_offer(grouped_scores, top_n=3)
    sem_pct = normalize_scores_to_percent(offer_scores)

    if not offer_scores:
        print(f"No offers found after filtering: {category_filter}")
        return

    # 3) LLM structuring: CV JSON once
    cv_json = extract_cv_json(cv_text)
    cv_skills = cv_json.get("skills", []) or []

    # 4) Build results list
    results = []
    for src, sc in offer_scores.items():
        offer_text = get_offer_full_text(jobs_db, src)
        job_json = extract_job_json(offer_text)

        offer_required = job_json.get("required_skills", []) or []
        ov = overlap_percent(cv_skills, offer_required)
        pct = final_percent(sem_pct.get(src, 0.0), ov, alpha_semantic=0.75)

        # evidence snippets using extracted skills/keywords
        kw = []
        kw += cv_skills[:10]
        kw += offer_required[:10]
        cv_snip = pick_snippet(cv_text, kw)
        offer_snip = pick_snippet(offer_text, kw)

        results.append({
            "source": src,
            "category": category_from_source(src),
            "semantic_avg_score": sc,
            "semantic_percent": sem_pct.get(src, 0.0),
            "overlap_percent": ov,
            "match_percent": pct,
            "cv_json": cv_json,       # same for all, repeated for convenience
            "job_json": job_json,
            "cv_snippet": cv_snip,
            "offer_snippet": offer_snip,
        })

    # 5) sort + keep top
    results.sort(key=lambda r: r["match_percent"], reverse=True)
    results = results[:top_offers]

    # 6) LLM explanations for top N
    for i, r in enumerate(results):
        if i < explain_top_n:
            r["llm_explanation"] = explain_match(
                r["cv_json"], r["job_json"], r["cv_snippet"], r["offer_snippet"]
            )
        else:
            r["llm_explanation"] = "(skipped to save time)"

    # 7) Pretty print
    print("\n" + "=" * 100)
    print(f"CV: {cv_filename}   | filter: {category_filter}")
    print("TOP OFFERS")
    print("=" * 100)

    print("\n--- CV JSON (structured) ---")
    print(r["cv_json"])  # same for all results

    for rank, r in enumerate(results, 1):
        print("\n" + "-" * 100)
        print(f"#{rank}  {os.path.basename(r['source'])}   [{r['category']}]")
        print(f"Match: {r['match_percent']:.1f}%  | semantic: {r['semantic_avg_score']:.4f} "
              f"(norm {r['semantic_percent']:.1f}%) | overlap(required): {r['overlap_percent']:.1f}%")

        print("\n--- Job JSON (structured) ---")
        print(r["job_json"])

        print("\n--- Evidence (CV) ---")
        print(f"\"{r['cv_snippet']}\"")

        print("\n--- Evidence (Offer) ---")
        print(f"\"{r['offer_snippet']}\"")

        print("\n--- LLM Explanation ---")
        print(r["llm_explanation"])

    return results


def list_cvs() -> List[str]:
    return sorted([f for f in os.listdir("CV") if f.lower().endswith(".pdf")])


if __name__ == "__main__":
    cvs = list_cvs()
    for cv in cvs:
        run_for_cv(cv, mode="auto", k_chunks=80, top_offers=5, explain_top_n=2)


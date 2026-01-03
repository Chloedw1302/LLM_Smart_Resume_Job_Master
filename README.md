# Smart Resume & Job Matcher

## Overview

This project implements an **AI-powered Resume and Job Matching System** based on **semantic embeddings and AI-driven reasoning**.
The system compares a candidate CV (PDF) with a job offer (text input) and outputs:

- A **match percentage**
- A **3-level verdict**:
  - ✅ Postuler (≥ 70%)
  - ⚠️ À tenter / Adapter le CV (50–70%)
  - ❌ Pas prioritaire (< 50%)
- An **AI-based explanation** including:
  - Alignment reasoning
  - Identified gaps
  - Actionable advice for the candidate

The goal is to go beyond keyword matching and rely on **language understanding** through embeddings.

---

## How AI Is Used in This Project

### 1. Semantic Understanding (Core AI Component)

- CVs and job offers are encoded using **Ollama embeddings** (`nomic-embed-text`)
- These embeddings capture:
  - Skills
  - Experience
  - Education
  - Contextual meaning
- Matching is performed using **semantic similarity**, not exact wording

➡️ Structured information (skills, experience, etc.) is **implicitly captured** in the embedding space rather than explicitly extracted into JSON fields.

---

### 2. AI-Driven Explanation Layer

The explanation for each match is **AI-driven but deterministic**:

- The **reasoning is powered by LLM embeddings**
- The **presentation is rule-based** for:
  - Speed
  - Stability
  - No hallucinations

This design choice ensures:
- Clear and consistent explanations
- Fast inference suitable for an interactive interface
- Reliable outputs for academic evaluation

---

## Project Structure

```
.
├── CV/                         # Input CVs (PDF only)
├── offres/                     # Job offers (.txt files)
├── db/                         # Chroma vector databases
├── app.py                      # Streamlit interface
├── ingestion_pipeline.py       # Vector DB creation (CVs & offers)
├── single_match_fast.py        # Fast CV ↔ offer matching logic
├── embeddings_ollama.py        # Ollama embedding wrapper
├── explain_rules.py            # Deterministic explanation logic
├── ollama_chat.py              # Ollama API interface (optional LLM usage)
├── match_pipeline.py           # Matching orchestration
├── llm_explain.py              # (Optional) LLM explanation experiments
├── llm_structuring.py          # (Optional) Structured extraction experiments
├── archive/                    # Previous or experimental pipelines
└── README.md
```

---


## How to Run the Project

### 1. Requirements

- Python 3.10+
- Ollama installed and running
- Ollama model installed:
```bash
ollama pull nomic-embed-text
```

---

### 2. Build the Vector Databases (once)

```bash
python ingestion_pipeline.py
```

This creates:
- CV embeddings
- Job offer embeddings
- Stored in ChromaDB (`db/`)

---

### 3. Run a Match in Terminal (No UI)

```bash
python single_match_fast.py
```

Outputs:
- Match %
- Verdict
- AI explanation

---

### 4. Run the Streamlit Interface

```bash
streamlit run app.py
```

Interface allows:
- Uploading a CV (PDF)
- Pasting or uploading a job offer
- Instant match result and explanation

---

## Output Example

```
Match %
75.9%
Verdict : ✅ Postuler
🧠 Analyse IA
✅ Match: compétences communes détectées (ai, data, power bi).
✅ Match: missions et expériences globalement compatibles selon l’analyse sémantique.
✅ Match: score de similarité élevé (75.9%).
⚠️ Gap: compétences attendues non mentionnées (python).
🎯 Conseil: ajouter ces compétences ou projets associés dans le CV.
```

---

## Authors / Notes

This project was designed as an academic AI system emphasizing **semantic understanding, explainability, and engineering trade-offs** rather than pure generative output.

DIA-3

Nour AFFES

Thomas VALESI

Chloé de WILDE

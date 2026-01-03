import json
from ollama_chat import ollama_generate

CV_SCHEMA = {
    "skills": [],
    "education": [],
    "experience": [],        # [{ "title": "", "company": "", "dates": "", "tasks": [] }]
    "certifications": [],
    "interests": []
}

JOB_SCHEMA = {
    "title": "",
    "required_skills": [],
    "nice_to_have": [],
    "missions": [],
    "domain": "",
    "location": ""
}

def _extract_json(text: str) -> dict:
    # tries to find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    return json.loads(text[start:end+1])

# ... dans llm_structuring.py

def extract_cv_json(cv_text: str) -> dict:
    prompt = f"""
You are an information extraction system.
Extract structured information from the CV text below.

Return ONLY valid JSON following this schema (same keys):
{json.dumps(CV_SCHEMA, ensure_ascii=False)}

Rules:
- skills: normalized (e.g., "Power BI", "SQL", "Python", "Pandas")
- experience: keep only the most relevant items, with tasks as short bullet strings
- if unknown, use empty lists

CV TEXT:
\"\"\"{cv_text[:2500]}\"\"\"
"""
    out = ollama_generate(prompt)
    data = _extract_json(out)
    return {**CV_SCHEMA, **data} if isinstance(data, dict) else CV_SCHEMA


def extract_job_json(job_text: str) -> dict:
    prompt = f"""
You are an information extraction system.
Extract structured information from the job description below.

Return ONLY valid JSON following this schema (same keys):
{json.dumps(JOB_SCHEMA, ensure_ascii=False)}

Rules:
- required_skills: skills/tools explicitly required
- nice_to_have: optional skills
- missions: short bullet strings
- domain: short label like "Data", "Finance", "Mechanical", etc.
- if unknown, keep empty string/list

JOB TEXT:
\"\"\"{job_text[:2500]}\"\"\"
"""
    out = ollama_generate(prompt)
    data = _extract_json(out)
    return {**JOB_SCHEMA, **data} if isinstance(data, dict) else JOB_SCHEMA

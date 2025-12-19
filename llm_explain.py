import json
from ollama_chat import ollama_generate

def explain_match(cv_json: dict, job_json: dict, cv_snip: str, job_snip: str) -> str:
    prompt = f"""
You are an AI career assistant. Explain why a candidate matches a job offer.

Candidate JSON:
{json.dumps(cv_json, ensure_ascii=False)}

Job JSON:
{json.dumps(job_json, ensure_ascii=False)}

Evidence from CV (quote):
\"\"\"{cv_snip}\"\"\"

Evidence from Job (quote):
\"\"\"{job_snip}\"\"\"

Write in French:
- 3 bullet points "✅ Match" (skills + experience + mission fit). Be specific.
- 1 bullet point "⚠️ Gap/Risque" (what might be missing).
- 1 bullet point "🎯 Conseil" (concrete improvement to CV/application).

No extra text outside bullet points.
"""
    return ollama_generate(prompt)

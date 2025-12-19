# explain_match.py
import json
from ollama_chat import ollama_chat

def generate_explanation(cv_json: dict, job_json: dict, cv_snip: str, job_snip: str) -> str:
    prompt = f"""
You are an AI career assistant. Explain why a candidate matches a job.

Candidate structured profile (JSON):
{json.dumps(cv_json, ensure_ascii=False)}

Job structured profile (JSON):
{json.dumps(job_json, ensure_ascii=False)}

Evidence from CV:
\"\"\"{cv_snip}\"\"\"

Evidence from Job:
\"\"\"{job_snip}\"\"\"

Write:
- 3 bullet points explaining the match (skills + experience + mission fit)
- 1 bullet point about a gap / risk
- 1 bullet point with a concrete suggestion to improve the application (CV wording or skill emphasis)

Be specific. Use the evidence snippets when possible.
"""
    return ollama_chat(prompt)

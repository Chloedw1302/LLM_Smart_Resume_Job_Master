# extract_structured.py
import json
from ollama_chat import ollama_chat

def extract_profile_json(text: str, kind: str = "cv") -> dict:
    """
    kind: "cv" or "job"
    """
    schema = {
        "skills": [],
        "education": [],
        "experience": [],   # list of objects {title, company, dates, tasks}
        "certifications": [],
        "interests": []
    }

    prompt = f"""
You are an information extraction system.
Extract structured information from the following {kind} text.

Return ONLY valid JSON following this schema keys:
{json.dumps(schema)}

Rules:
- Keep it concise.
- Skills must be normalized (e.g., "Power BI", "SQL", "Python").
- Experience: include title + main tasks (bullet-like strings).
- If a field is unknown, return an empty list.

TEXT:
\"\"\"{text[:6000]}\"\"\"
"""
    out = ollama_chat(prompt)
    # Try to parse JSON even if model adds text
    start = out.find("{")
    end = out.rfind("}")
    if start == -1 or end == -1:
        return schema
    return json.loads(out[start:end+1])

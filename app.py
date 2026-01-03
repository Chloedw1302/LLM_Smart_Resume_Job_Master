import os
import tempfile
import streamlit as st
from single_match_fast import single_match_fast

st.set_page_config(page_title="Smart Resume & Job Matcher", page_icon="🧠")

st.title("🧠 Smart Resume & Job Matcher")
st.write("Dépose un **CV (PDF)** et une **offre**. Le système te dit si ça vaut le coup de postuler.")

cv_file = st.file_uploader("📄 CV (PDF)", type=["pdf"])

st.subheader("🧾 Offre de stage")
mode = st.radio(
    "Format de l’offre",
    ["Uploader un .txt", "Copier / coller le texte"],
    horizontal=True
)

offer_text = ""

if mode == "Uploader un .txt":
    offer_file = st.file_uploader("Offre (.txt)", type=["txt"])
    if offer_file:
        offer_text = offer_file.read().decode("utf-8", errors="ignore")
else:
    offer_text = st.text_area(
        "Colle ici le texte de l’offre",
        height=220,
        placeholder="Colle le texte complet de l'offre ici..."
    )

run = st.button(
    "🚀 Lancer le matching",
    type="primary",
    disabled=(cv_file is None or len(offer_text.strip()) < 100)
)

if run:
    with st.spinner("Analyse en cours..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(cv_file.read())
            cv_path = tmp.name

        try:
            result = single_match_fast(cv_path, offer_text)
        finally:
            os.remove(cv_path)

    st.success("Analyse terminée ✅")

    st.metric("Match %", f"{result['match_percent']:.1f}%")
    st.write(f"### Verdict : {result['verdict']}")

    st.subheader("🧠 Analyse IA")
    st.write(result["llm_explanation"])
    #st.code(result["llm_explanation"])
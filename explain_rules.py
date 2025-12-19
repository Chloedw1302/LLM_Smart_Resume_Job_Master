KEYWORDS_DATA = [
    "python", "sql", "data", "machine learning", "ml", "ai",
    "pandas", "numpy", "power bi", "tableau", "dashboard",
    "erp", "cloud", "nlp"
]

def extract_keywords(text):
    low = text.lower()
    return {k for k in KEYWORDS_DATA if k in low}


def explain_match_rules(cv_text, offer_text, match_percent):
    cv_k = extract_keywords(cv_text)
    offer_k = extract_keywords(offer_text)

    common = sorted(cv_k & offer_k)
    missing = sorted(offer_k - cv_k)

    bullets = []

    if common:
        bullets.append(f"- ✅ Match: compétences communes détectées ({', '.join(common[:3])}).")
    else:
        bullets.append("- ✅ Match: adéquation sémantique globale entre le CV et l’offre.")

    bullets.append("- ✅ Match: missions et expériences globalement compatibles selon l’analyse sémantique.")

    bullets.append(f"- ✅ Match: score de similarité élevé ({match_percent:.1f}%).")

    if missing:
        bullets.append(f"- ⚠️ Gap: compétences attendues non mentionnées ({', '.join(missing[:3])}).")
        bullets.append("- 🎯 Conseil: ajouter ces compétences ou projets associés dans le CV.")
    else:
        bullets.append("- ⚠️ Gap: aucun manque majeur détecté.")
        bullets.append("- 🎯 Conseil: postuler en mettant en avant les projets les plus pertinents.")

    return "\n".join(bullets[:5])
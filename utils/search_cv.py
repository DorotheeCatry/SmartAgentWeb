import json
from langchain.prompts import PromptTemplate
from core.llm_providers import LLMManager

# Charger la base de données des CV
with open("data/cv_data_extended.json", "r", encoding="utf-8") as f:
    CV_DATABASE = json.load(f)

# Définir le prompt pour l'extraction des critères
CRITERIA_EXTRACTION_PROMPT = PromptTemplate.from_template("""
Tu es un recruteur RH expert. Analyse cette demande :
{query}
Donne des conseils de sourcing :
- Type de contrat recommandé (alternance, sous-traitance, CDD...),
- Délai moyen de recrutement pour ces profils,
- Difficultés éventuelles.
Réponds sous ce format JSON :
{{
"recommandation": "texte synthétique"
}}
""")

llm = LLMManager().get_llm()

def parse_user_query(query: str) -> dict:
    prompt_text = CRITERIA_EXTRACTION_PROMPT.format_prompt(query=query).to_string()
    response = llm.invoke(prompt_text)
    try:
        criteria = json.loads(response)
    except Exception:
        criteria = {}
    return criteria

def search_cv_local(criteria: dict) -> list:
    mots_cles = criteria.get("mots_cles", [])
    localisation = criteria.get("localisation")
    niveau = criteria.get("experience_niveau")
    statut = criteria.get("statut")
    dispo = criteria.get("disponibilite")
    results = []
    for cv in CV_DATABASE:
        if mots_cles and not any(c.lower() in map(str.lower, cv["competences"]) for c in mots_cles):
            continue
        if localisation and localisation.lower() not in cv["localisation"].lower() and localisation != cv["code_postal"]:
            continue
        if niveau and niveau.lower() != cv.get("experience_niveau", "").lower():
            continue
        if statut and statut.lower() != cv.get("statut", "").lower():
            continue
        if dispo and dispo.lower() != cv.get("disponibilite", "").lower():
            continue
        results.append(cv)
    return results

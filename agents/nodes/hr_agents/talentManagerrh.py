from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any, Literal
import json
import logging

class TalentManagerAgent(Runnable):
    def __init__(self):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retourne un dictionnaire validé contenant le plan de gestion des talents.
        Garantit toujours le même format de sortie même en cas d'erreur.
        """
        query = str(input.get("query", ""))[:2000]  # Limite la longueur
        
        try:
            # Prompt structuré avec validation stricte
            prompt = f"""### Rôle ###
Expert en gestion des talents - Format JSON strict

### Projet ###
{query}

### Consignes ###
Analyser et répondre en JSON VALIDE avec ces champs EXACTS :
1. upskill_possible (bool)
2. duree_formation (format: "X jours/semaines/mois")
3. referentiel_competences ("oui" ou "non")
4. plan_global (3-5 points max)

### Exemple de Réponse ###
{{
    "upskill_possible": true,
    "duree_formation": "3 semaines",
    "referentiel_competences": "oui",
    "plan_global": "1. Identifier les compétences clés\n2. Évaluer les écarts\n3. Planifier les formations"
}}"""

            # Appel LLM
            response = self.llm.invoke(prompt)
            
            # Parsing et validation
            return self._validate_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur TalentManager: {str(e)}")
            return self._default_response(str(e))

    def _validate_response(self, response: Any) -> Dict[str, Any]:
        """Valide et transforme la réponse du LLM"""
        # Conversion si string JSON
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                self.logger.warning("Réponse JSON invalide")
                return self._default_response("Format JSON invalide")

        # Vérification du type
        if not isinstance(response, dict):
            return self._default_response("Type de réponse invalide")

        # Validation des champs
        validated = {
            "upskill_possible": bool(response.get("upskill_possible", False)),
            "duree_formation": str(response.get("duree_formation", "Non estimable"))[:50],
            "referentiel_competences": "oui" if str(response.get("referentiel_competences", "")).lower() == "oui" else "non",
            "plan_global": str(response.get("plan_global", ""))[:500]
        }
        
        return validated

    def _default_response(self, error_msg: str = "") -> Dict[str, Any]:
        """Réponse par défaut en cas d'erreur"""
        return {
            "upskill_possible": False,
            "duree_formation": "Non estimable",
            "referentiel_competences": "oui",
            "plan_global": f"Erreur: {error_msg[:200]}" if error_msg else "Analyse indisponible"
        }

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
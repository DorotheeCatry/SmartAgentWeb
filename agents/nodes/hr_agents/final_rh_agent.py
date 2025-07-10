from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any
import json
import logging

class FinalRHAgent(Runnable):
    def __init__(self):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extraction sécurisée des inputs
            answers = input.get("answers", {})
            critiques = input.get("critiques", {})
            validations = input.get("validations", {})
            
            # Conversion en string pour le prompt
            answers_str = json.dumps(answers) if isinstance(answers, dict) else str(answers)
            critiques_str = json.dumps(critiques) if isinstance(critiques, dict) else str(critiques)
            validations_str = json.dumps(validations) if isinstance(validations, dict) else str(validations)
            
            prompt = f"""Tu es un expert RH qui fait une synthèse finale. Analyse ces données et réponds UNIQUEMENT avec un JSON valide :

Réponses des agents : {answers_str[:1500]}
Critiques : {critiques_str[:1000]}
Validations : {validations_str[:1000]}

Réponds STRICTEMENT avec ce format JSON (sans texte avant ou après) :
{{
    "faisabilite": "Oui",
    "conditions_reussite": ["condition1", "condition2"],
    "score_confiance": 0.8,
    "recommandation": "recommandation détaillée",
    "risques_principaux": ["risque1", "risque2"]
}}"""
            
            response = self.llm.invoke(prompt)
            return self._format_final_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur FinalRHAgent: {str(e)}")
            return self._error_response(str(e))

    def _format_final_response(self, response: str) -> Dict[str, Any]:
        """Formatage robuste de la réponse finale"""
        try:
            # Nettoyage de la réponse
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            
            # Extraction du JSON
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                data = json.loads(json_str)
                
                # Validation et conversion des types
                faisabilite = str(data.get("faisabilite", "Indéterminé"))
                if faisabilite not in ["Oui", "Non", "Partiel"]:
                    faisabilite = "Indéterminé"
                
                return {
                    "faisabilite": faisabilite,
                    "conditions_reussite": [str(c)[:100] for c in data.get("conditions_reussite", [])][:5],
                    "score_confiance": min(max(float(data.get("score_confiance", 0.5)), 0.0), 1.0),
                    "recommandation": str(data.get("recommandation", ""))[:500],
                    "risques_principaux": [str(r)[:100] for r in data.get("risques_principaux", [])][:5]
                }
            else:
                raise ValueError("Pas de JSON trouvé")
                
        except Exception as e:
            self.logger.warning(f"Erreur parsing final: {str(e)}")
            return self._error_response(f"Erreur de format: {str(e)}")

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "faisabilite": "Erreur",
            "conditions_reussite": ["Vérifier les logs système"],
            "score_confiance": 0.0,
            "recommandation": f"Erreur de traitement: {error[:200]}",
            "risques_principaux": ["Erreur technique dans l'analyse"]
        }
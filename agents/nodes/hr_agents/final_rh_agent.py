from langchain_core.runnables import Runnable
from agents.nodes.base_agent import BaseAgent
from core.llm_providers import LLMManager
from typing import Dict, Any, List
import json
import logging
from utils.json_utils import JSONRepairer

class FinalRHAgent(Runnable):
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extraction sécurisée des inputs
            answers = input.get("answers", {})
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers)
                except:
                    answers = {"error": "Invalid answers format"}
                    
            critiques = input.get("critiques", {})
            validations = input.get("validations", {})
            
            prompt = f"""
            Synthèse finale - Format JSON strict
            
            Réponses: {json.dumps(answers)[:1000]}
            Critiques: {json.dumps(critiques)[:1000]}
            Validations: {json.dumps(validations)[:1000]}
            
            Format de sortie REQUIS:
            {{
                "faisabilite": "Oui|Non|Partiel",
                "conditions_reussite": ["liste"],
                "score_confiance": 0.0-1.0,
                "recommandation": "texte",
                "risques_principaux": ["liste"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            return self._format_final_response(response)
            
        except Exception as e:
            return self._error_response(str(e))

    def _format_final_response(self, response: str) -> Dict[str, Any]:
        """Formatage robuste de la réponse finale"""
        try:
            # Nettoyage et parsing
            response = response.replace('```json', '').replace('```', '').strip()
            data = json.loads(response.split('{', 1)[-1].split('}', 1)[0].join('{}'))
            
            # Validation et conversion des types
            return {
                "faisabilite": str(data.get("faisabilite", "Indéterminé")),
                "conditions_reussite": list(data.get("conditions_reussite", [])),
                "score_confiance": min(max(float(data.get("score_confiance", 0)), 0), 1),
                "recommandation": str(data.get("recommandation", ""))[:500],
                "risques_principaux": list(data.get("risques_principaux", []))
            }
        except Exception as e:
            return self._error_response(f"Erreur de format: {str(e)}")

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "faisabilite": "Erreur",
            "conditions_reussite": ["Vérifier les logs"],
            "score_confiance": 0.0,
            "recommandation": error[:200],
            "risques_principaux": ["Erreur technique"]
        }
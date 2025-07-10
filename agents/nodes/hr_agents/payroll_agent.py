from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any
import json
import logging

class PayrollAgent(Runnable):
    def __init__(self):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
        # Valeurs par défaut en cas d'erreur
        self._default_response = {
            "cout_mensuel": "erreur",
            "budget_suffisant": "non",
            "analyse_marche": "Erreur d'analyse",
            "recommandations": ["Contacter le service RH"]
        }

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue les aspects financiers d'un projet RH avec :
        - Gestion robuste des erreurs
        - Validation du format de sortie
        - Limitation de la taille des entrées/sorties
        """
        try:
            query = str(input.get("query", ""))[:500]  # Limite la longueur
            
            prompt = f"""ANALYSE PAIE - FORMAT JSON STRICT
            Demande : {query}
            
            Consignes :
            1. Coût mensuel (ex: "3 500€")
            2. Budget suffisant ("oui"/"non")
            3. Analyse marché (30 mots max)
            4. 3-5 recommandations
            
            Modèle de réponse :
            {{
                "cout_mensuel": "X€",
                "budget_suffisant": "oui/non",
                "analyse_marche": "texte synthétique",
                "recommandations": ["item1", "item2"]
            }}"""
            
            # Appel LLM sécurisé
            raw_response = self.llm.invoke(prompt)
            response = self._parse_response(raw_response)
            
            # Validation finale
            if not self._validate_response(response):
                raise ValueError("Format de réponse invalide")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur PayrollAgent: {str(e)}")
            return self._default_response

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Transforme la réponse en JSON valide"""
        if isinstance(response, dict):
            return response
        try:
            return json.loads(str(response))
        except json.JSONDecodeError:
            # Fallback pour réponses mal formatées
            return {
                "cout_mensuel": str(response)[:100],
                "budget_suffisant": "non",
                "analyse_marche": "Réponse non structurée",
                "recommandations": ["Analyser manuellement"]
            }

    def _validate_response(self, response: Dict) -> bool:
        """Valide le format de la réponse"""
        required_keys = {
            "cout_mensuel": str,
            "budget_suffisant": str,
            "recommandations": list
        }
        
        for key, typ in required_keys.items():
            if key not in response or not isinstance(response[key], typ):
                return False
                
        # Validation des valeurs permises
        if response["budget_suffisant"].lower() not in ("oui", "non"):
            return False
            
        return True

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
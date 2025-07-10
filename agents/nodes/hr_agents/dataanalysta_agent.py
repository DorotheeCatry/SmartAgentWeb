from typing import Dict, Any
import json
import logging
from langchain_core.runnables import Runnable

class DataAnalystAgent(Runnable):
    def __init__(self, retriever):
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = str(input.get("query", ""))
            docs = self.retriever.invoke(query)
            context = "\n".join(d.page_content for d in docs[:3])
            
            prompt = f"""
            Tu es un data analyst RH. Analyse cette demande :
            {query[:1000]}
            
            Contexte :
            {context[:2000]}
            
            Réponds STRICTEMENT en JSON valide avec ces champs :
            {{
                "bassin_emploi": "texte",
                "disponibilite_profils": "texte", 
                "tendances_marche": ["liste"],
                "erreur": null
            }}
            """
            
            response = self.llm.invoke(prompt)
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur DataAnalyst: {str(e)}")
            return {
                "bassin_emploi": "Erreur",
                "disponibilite_profils": "Erreur",
                "tendances_marche": [],
                "erreur": str(e)
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse robuste des réponses JSON"""
        try:
            # Nettoyage initial
            cleaned = response.replace('```json', '').replace('```', '').strip()
            
            # Extraction du JSON même s'il est entouré de texte
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            json_str = cleaned[start:end]
            
            data = json.loads(json_str)
            
            # Validation des champs requis
            required = ["bassin_emploi", "disponibilite_profils", "tendances_marche"]
            if not all(key in data for key in required):
                raise ValueError("Champs manquants dans la réponse")
                
            return data
            
        except Exception as e:
            self.logger.warning(f"Réponse JSON invalide: {response[:200]}")
            return {
                "bassin_emploi": "Données indisponibles",
                "disponibilite_profils": "Données indisponibles",
                "tendanes_marche": [],
                "erreur": str(e)
            }
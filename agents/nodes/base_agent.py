from typing import Any, Dict
from langchain_core.runnables import Runnable
import logging
from utils.json_utils import JSONRepairer

class BaseAgent(Runnable):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.json_utils = JSONRepairer()
        
    def process_response(self, response: Any, required_fields: list) -> Dict:
        """Méthode standardisée de traitement des réponses"""
        try:
            # Conversion en dict si nécessaire
            if isinstance(response, str):
                response = self.json_utils.safe_parse(response)
            
            # Validation de la structure
            if not self.json_utils.validate_structure(response, required_fields):
                raise ValueError("Structure JSON invalide")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur de traitement: {str(e)}")
            return {"error": str(e), "raw_response": str(response)[:200]}
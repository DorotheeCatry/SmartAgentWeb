from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

class RHResponse(BaseModel):
    """Modèle de réponse standardisé pour l'agent RH"""
    response: str
    legal_references: Dict[str, str]
    compliance_status: str
    error: Optional[bool] = False
    error_details: Optional[str] = None

class RHAgent(Runnable):
    def __init__(self):
        self.llm = LLMManager().get_llm()
        self.max_input_length = 2000  # Limite de caractères pour l'input
        self.max_output_length = 5000  # Limite de caractères pour l'output

        self.system_prompt = """[SYSTEM]
        Expert en droit du travail français - Réponses structurées
        Exige le format JSON suivant :
        {
            "response": "texte concis",
            "legal_references": {
                "code_du_travail": "article",
                "convention_collective": "IDCC"
            },
            "compliance_status": "conforme|à_verifier|non_conforme"
        }"""

    def _truncate_input(self, text: str) -> str:
        """Garantit que l'input ne dépasse pas la taille maximale"""
        return text[:self.max_input_length]

    def _validate_output(self, data: Dict) -> Dict:
        """Valide et nettoie la sortie du LLM"""
        if not isinstance(data, dict):
            raise ValueError("Réponse non valide: format JSON attendu")
            
        return {
            "response": str(data.get("response", ""))[:self.max_output_length],
            "legal_references": {
                "code_du_travail": str(data.get("legal_references", {}).get("code_du_travail", "À compléter")),
                "convention_collective": str(data.get("legal_references", {}).get("convention_collective", "À compléter"))
            },
            "compliance_status": str(data.get("compliance_status", "à_verifier")).lower()
        }

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Version robuste avec gestion d'erreurs et limites de taille
        """
        try:
            # Validation de l'input
            question = self._truncate_input(str(input.get("query", "")))
            if not question.strip():
                raise ValueError("Question vide")
            
            # Construction du prompt
            prompt = f"""{self.system_prompt}
            
            [QUESTION]
            {question}
            
            [CONTEXTE]
            {json.dumps(input.get('context', {}), ensure_ascii=False)[:1000]}
            
            [EXIGENCES]
            - Réponse en JSON VALIDE
            - Maximum {self.max_output_length} caractères
            - Citer au moins 1 article de loi"""
            
            # Appel LLM
            raw_response = self.llm.invoke(prompt)
            
            # Parsing et validation
            if isinstance(raw_response, str):
                try:
                    response_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.warning("Réponse JSON invalide, tentative de parsing manuel")
                    response_data = {"response": raw_response[:self.max_output_length]}
            else:
                response_data = raw_response
                
            validated_data = self._validate_output(response_data)
            
            return RHResponse(
                **validated_data
            ).dict()
            
        except Exception as e:
            logger.error(f"Erreur RHAgent: {str(e)}")
            return RHResponse(
                response=f"Erreur: {str(e)[:200]}",
                legal_references={},
                compliance_status="non_conforme",
                error=True,
                error_details=str(e)
            ).dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Version asynchrone avec les mêmes garanties"""
        return self.invoke(input)
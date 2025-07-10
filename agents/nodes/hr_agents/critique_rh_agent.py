from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any, List
import json
import logging

class CritiqueRHAgent(Runnable):
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = input.get("content", "")
            if not content:
                return self._empty_response()
                
            prompt = f"""
            Critique RH - Format JSON strict
            
            Contenu:
            {content[:5000]}
            
            Format de réponse REQUIS:
            {{
                "points_forts": ["liste"],
                "points_faibles": ["liste"],
                "suggestions": ["liste"],
                "note_coherence": 0-5,
                "commentaire_global": "texte"
            }}
            """
            
            response = self.llm.invoke(prompt)
            return self._parse_critique(response)
            
        except Exception as e:
            return self._error_response(str(e))

    def _parse_critique(self, response: str) -> Dict[str, Any]:
        """Parsing robuste des critiques"""
        try:
            # Extraction du JSON même malformé
            json_str = response[response.find('{'):response.rfind('}')+1]
            data = json.loads(json_str)
            
            return {
                "points_forts": list(data.get("points_forts", [])),
                "points_faibles": list(data.get("points_faibles", [])),
                "suggestions": list(data.get("suggestions", [])),
                "note_coherence": min(max(int(data.get("note_coherence", 0)), 0), 5),
                "commentaire_global": str(data.get("commentaire_global", ""))[:200]
            }
        except:
            return self._error_response("Format de critique invalide")
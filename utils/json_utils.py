import json
import re
from typing import Any, Dict

import json
import re
from typing import Any, Dict

class JSONRepairer:
    @staticmethod
    def repair(json_str: str) -> Dict[str, Any]:
        """Réparation agressive des JSON malformés"""
        try:
            # Nettoyage initial
            json_str = json_str.strip()
            
            # Suppression des blocs de code
            json_str = re.sub(r'^```(json)?|```$', '', json_str, flags=re.IGNORECASE)
            
            # Extraction du contenu JSON
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if not match:
                raise ValueError("Aucun JSON valide trouvé")
                
            json_content = match.group(0)
            
            # Correction des erreurs courantes
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)  # Virgules traînantes
            json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)  # Caractères non-printables
            
            return json.loads(json_content)
            
        except json.JSONDecodeError:
            # Fallback: extraction manuelle des paires clé-valeur
            try:
                pairs = re.findall(r'"([^"]+)"\s*:\s*("[^"]+"|\d+|true|false|null)', json_str)
                return {k: json.loads(v) if v.startswith('"') else eval(v) for k, v in pairs}
            except:
                return {"error": "Échec de réparation JSON"}
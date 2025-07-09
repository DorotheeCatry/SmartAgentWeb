from typing import Dict, Any, Optional
import re
import json


class ActionParser:
    """
    Parseur d'actions depuis des chaînes ou JSON.
    Transforme des instructions textuelles ou JSON en actions standardisées.
    """

    def parse(self, input_text: str) -> Optional[Dict[str, Any]]:
        """
        Analyse un texte libre pour extraire une action et ses paramètres.

        Args:
            input_text (str): Texte ou JSON à parser.

        Returns:
            dict ou None: Dictionnaire contenant 'action' et 'parameters', ou None si non reconnu.
        """
        # Tentative d'interprétation JSON
        try:
            data = json.loads(input_text)
            if "action" in data and "parameters" in data:
                return {
                    "action": data["action"],
                    "parameters": data["parameters"]
                }
        except json.JSONDecodeError:
            pass

        # Exemple simple de parsing texte (basique)
        # On cherche une action simple dans la phrase
        # Exemple : "search weather in Paris"
        patterns = {
            "search_weather": r"(search|find|get).*(weather|temperature).*(in|at)? (?P<location>\w+)",
            "search_news": r"(search|find|get).*(news).*(about|on)? (?P<topic>\w+)",
            # Ajouter d'autres patterns ici
        }

        for action, pattern in patterns.items():
            match = re.search(pattern, input_text, re.IGNORECASE)
            if match:
                params = match.groupdict()
                # Nettoyage des paramètres, retirer None
                params = {k: v for k, v in params.items() if v}
                return {
                    "action": action,
                    "parameters": params
                }

        # Si aucun pattern ne matche, retourner None
        return None

from typing import Any, Dict, Optional

class AgentState:
    """
    Classe pour stocker et gérer l'état contextuel / mémoire partagée
    entre les nœuds LangGraph pendant une session d'interaction.

    Cette classe peut être étendue pour ajouter des fonctionnalités
    avancées (ex: historique, nettoyage, accès facilité, etc.).
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        # Initialise un dictionnaire interne pour la mémoire partagée
        self._memory: Dict[str, Any] = initial_state or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère la valeur associée à la clé dans la mémoire.
        """
        return self._memory.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Stocke une valeur dans la mémoire sous la clé donnée.
        """
        self._memory[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """
        Met à jour la mémoire avec un dictionnaire de données.
        """
        self._memory.update(data)

    def clear(self) -> None:
        """
        Vide la mémoire.
        """
        self._memory.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Retourne une copie de la mémoire sous forme de dictionnaire.
        """
        return dict(self._memory)

    def __repr__(self) -> str:
        return f"<AgentState memory={self._memory}>"

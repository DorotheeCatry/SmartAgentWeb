from agents.graphs.smart_graph import create_smart_graph

class AgentManager:
    """
    Gestionnaire central des agents via le graphe LangGraph.
    """

    def __init__(self):
        self.graph = create_smart_graph().compile()

    def handle_query(self, query: str):
        """
        Point d'entrée pour traiter une requête utilisateur.
        """
        input_data = {"query": query}
        result = self.graph.invoke(input_data)
        return result


# Exemple d'utilisation rapide
if __name__ == "__main__":
    manager = AgentManager()
    query = "Quels sont les bienfaits du thé vert ?"
    response = manager.handle_query(query)
    print("Réponse finale de l'agent manager :", response)
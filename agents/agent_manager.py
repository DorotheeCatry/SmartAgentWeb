from agents.graphs.smart_graph import create_smart_graph
from agents.vector.retrieve import create_dynamic_rh_graph

class AgentManager:
    """
    Gestionnaire central des agents via le graphe LangGraph.
    """

    def __init__(self, mode="smart"):
        if mode == "smart":
            self.graph = create_smart_graph().compile()
        elif mode == "rh":
            self.graph = create_dynamic_rh_graph().compile()

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
    query = "Est-ce que IA peut frein au développement dans le monde entier, vu qu'il y a beaucoup de chomage "
    #"Est ce qu'il y a un problème lorsqu'on boit du thé vert ?"
    response = manager.handle_query(query)
    print("Réponse finale de l'agent manager :", response)
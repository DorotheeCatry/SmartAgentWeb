from typing import Dict
from langgraph.graph import StateGraph, END

from agents.nodes.web_agents.search_web_agent import SearchWebAgent
from agents.nodes.web_agents.critique_agent import CritiqueAgent
from agents.nodes.web_agents.validation_agent import ValidationAgent

# Agents spécialisés
search_agent = SearchWebAgent()
critique_agent = CritiqueAgent()
validation_agent = ValidationAgent()

# === Fonctions de nœuds ===

def node_search(state: dict) -> dict:
    query = state.get("query", "")
    output = search_agent.invoke({"query": query})
    state["search_results"] = output.get("summary") or output.get("results") or ""
    return state

def node_critique(state: dict) -> dict:
    content = state.get("search_results", "")
    output = critique_agent.invoke({"content": content})
    state["critique"] = output.get("critique", "")
    return state

def node_validation(state: dict) -> dict:
    content = state.get("search_results", "")
    output = validation_agent.invoke({"content": content})
    state["validation"] = output.get("validation", "")
    return state

# === Fonctions de routage conditionnel ===

def route_after_search(state: dict) -> str:
    return "critique"

def route_after_critique(state: dict) -> str:
    critique = state.get("critique", "").lower()
    return "validation" if "problème" in critique else END

def route_after_validation(state: dict) -> str:
    return END

# === Création du graphe ===

def create_smart_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("search", node_search)
    graph.add_node("critique", node_critique)
    graph.add_node("validation", node_validation)

    graph.set_entry_point("search")

    graph.add_conditional_edges("search", route_after_search, {"critique": "critique"})
    graph.add_conditional_edges("critique", route_after_critique, {"validation": "validation", END: END})
    graph.add_conditional_edges("validation", route_after_validation, {END: END})

    return graph

# === Test exécutable ===

if __name__ == "__main__":
    graph = create_smart_graph().compile()
    result = graph.invoke({"query": "Quels sont les bienfaits du thé vert ?"})
    print("Résultat final :", result)

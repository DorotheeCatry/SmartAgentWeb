from typing import Literal
from agents.state.agent_state import AgentState
from core.llm_providers import LLMManager

def complexity_router(state: AgentState) -> Literal["simple_path", "reflection_path"]:
    """
    Utilise un LLM pour déterminer si la tâche est simple ou complexe.
    La décision est enregistrée dans l'état (state) pour les étapes suivantes.
    """
    query = state.get("query", "")

    # Prompt système pour classifier
    prompt = f"""
    Tu es un assistant qui aide à déterminer la complexité d'une tâche pour un agent conversationnel.
    Si la tâche peut être résolue directement avec une seule réponse simple ou recherche web → réponds "simple".
    Si la tâche nécessite plusieurs étapes de raisonnement, une validation, ou une réflexion critique → réponds "complex".

    Exemple de tâche: {query}
    Complexité (réponds uniquement par "simple" ou "complex") :
    """.strip()

    # Utilisation du LLM (Ollama ou Groq)
    llm = LLMManager().get_llm()
    response = llm.invoke(prompt).lower().strip()

    # Par sécurité, fallback si réponse floue
    if "complex" in response:
        complexity = "complex"
        next_node = "reflection_path"
    else:
        complexity = "simple"
        next_node = "simple_path"

    # ✅ Mise à jour correcte de l'état
    state["complexity"] = complexity

    return next_node

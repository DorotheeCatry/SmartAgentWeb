# agents/nodes/response_generation.py

from core.llm_providers import LLMManager

def node_generate_response(state: dict) -> dict:
    """
    Génère une réponse simple et directe à partir des résultats de recherche.
    """
    query = state.get("query", "")
    search_result = state.get("search_results", "")

    prompt = f"""Tu es un assistant expert.
    Tu dois répondre de manière claire, concise et utile à la question suivante en utilisant les informations ci-dessous.
    Question : {query}
    Infos : {search_result}
    Réponse :"""

    llm = LLMManager().get_llm()
    answer = llm.invoke(prompt)

    state["final_response"] = answer.strip()
    return state

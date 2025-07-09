# search_web_agent.py

from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema.runnable import Runnable
from langchain_core.runnables import RunnableMap
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_providers import LLMManager

# Outil de recherche web
search_tool = DuckDuckGoSearchRun()

# Gestionnaire LLM (Groq / Ollama)
llm = LLMManager().get_llm()

# Prompt pour résumer les résultats
SUMMARY_PROMPT = PromptTemplate.from_template("""
Tu es un assistant de recherche. Voici des résultats de recherche Web :
{results}

Résume-les en une réponse utile pour l'utilisateur (en français, concise, neutre et utile).
""")

# Étape 1 : rechercher
def search_web(query: str) -> str:
    print(f"🔍 Recherche web pour : {query}")
    return search_tool.run(query)

# Étape 2 : résumer avec LLM
def summarize_results(results: str) -> str:
    prompt = SUMMARY_PROMPT.format(results=results)
    return llm.invoke(prompt)

# Composant LangGraph : agent combiné
class SearchWebAgent(Runnable):
    """Agent combiné : Recherche web + Résumé LLM"""

    def invoke(self, input: dict) -> dict:
        query = input["query"]
        web_results = search_web(query)
        summary = summarize_results(web_results)
        return {
            "query": query,
            "results": web_results,
            "summary": summary
        }

    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)  # Pour compatibilité asynchrone LangGraph

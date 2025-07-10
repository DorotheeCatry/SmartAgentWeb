from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.dataanalysta_agent import DataAnalystAgent
from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent
from agents.nodes.hr_agents.rhagent import RHAgent
from agents.nodes.hr_agents.talentManagerrh import TalentManagerAgent
from agents.nodes.hr_agents.onboarding_agent import OnboardingAgent
from agents.nodes.hr_agents.payroll_agent import PayrollAgent
from agents.nodes.hr_agents.critique_rh_agent import CritiqueRHAgent
from agents.nodes.hr_agents.validation_rh_agent import ValidationRHAgent
from agents.nodes.hr_agents.final_rh_agent import FinalRHAgent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging
from utils.json_utils import JSONRepairer

# Configuration du logger
logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"

def get_local_retriever():
    """Initialise et retourne le retriever Chroma"""
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        ).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
        raise

def create_project_graph() -> StateGraph:
    """Crée et configure le graphe de traitement RH avec gestion robuste des erreurs"""
    graph = StateGraph(GraphState)
    
    # Initialisation des agents avec gestion des erreurs
    try:
        retriever = get_local_retriever()
        dataanalyst = DataAnalystAgent(retriever=retriever)
        recruiter = RecruiterAgent()
        rh = RHAgent()
        talent = TalentManagerAgent()
        onboarding = OnboardingAgent()
        payroll = PayrollAgent()
        critique = CritiqueRHAgent()
        validation = ValidationRHAgent()
        final = FinalRHAgent()
    except Exception as e:
        logger.critical(f"Erreur d'initialisation des agents: {str(e)}")
        raise

    def safe_node_wrapper(agent, key: str, needs_data: bool = False):
        """Wrapper sécurisé pour les nœuds avec gestion des erreurs"""
        def node(state: GraphState) -> Dict[str, Any]:
            try:
                input_data = {"query": state["query"]}
                
                if needs_data and "data_analytics" in state:
                    input_data["data"] = state["data_analytics"]
                
                result = agent.invoke(input_data)
                
                # Validation et réparation du JSON
                if isinstance(result, str):
                    result = JSONRepairer.repair(result)
                
                return {key: result}
                
            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                return {
                    key: {
                        "error": str(e),
                        "stack_trace": f"{type(e).__name__}: {str(e)}"
                    }
                }
        return node

    def critique_node(state: GraphState) -> Dict[str, Any]:
        """Nœud de critique avec validation des données"""
        try:
            content_parts = []
            required_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]
            
            for node in required_nodes:
                if node in state:
                    content = str(state[node])
                    if len(content) > 2000:  # Limite de taille
                        content = content[:1000] + " [...] " + content[-1000:]
                    content_parts.append(f"{node.upper()}:\n{content}")
            
            if not content_parts:
                return {"critique": {"error": "Aucune donnée à analyser"}}
            
            critique_input = {"content": "\n\n".join(content_parts)[:10000]}  # Limite totale
            return {"critique": critique.invoke(critique_input)}
            
        except Exception as e:
            logger.error(f"Erreur dans critique_node: {str(e)}")
            return {"critique": {"error": str(e)}}

    def validation_node(state: GraphState) -> Dict[str, Any]:
        """Nœud de validation avec parsing sécurisé"""
        try:
            critique_content = state.get("critique", {})
            if isinstance(critique_content, str):
                critique_content = JSONRepairer.repair(critique_content)
                
            if not critique_content or "error" in critique_content:
                return {"validation": {"validation": "non valide", "justification": "Critique invalide"}}
                
            validation_result = validation.invoke({"critique": critique_content})
            
            # Post-processing pour garantir le format
            if isinstance(validation_result, str):
                validation_result = JSONRepairer.repair(validation_result)
                
            return {"validation": validation_result}
            
        except Exception as e:
            logger.error(f"Erreur dans validation_node: {str(e)}")
            return {"validation": {
                "validation": "erreur",
                "justification": str(e)[:200]
            }}

    def final_node(state: GraphState) -> Dict[str, Any]:
        """Nœud final avec agrégation sécurisée"""
        try:
            inputs = {
                "answers": "\n".join(
                    str(state.get(k, "")) 
                    for k in ["recruiter", "rh", "talent", "onboarding", "payroll"]
                ),
                "critiques": state.get("critique", {}),
                "validations": state.get("validation", {})
            }
            
            # Nettoyage des inputs
            for key in inputs:
                if isinstance(inputs[key], str):
                    inputs[key] = JSONRepairer.repair(inputs[key])
            
            return {"final_answer": final.invoke(inputs)}
            
        except Exception as e:
            logger.error(f"Erreur dans final_node: {str(e)}")
            return {"final_answer": {
                "error": str(e),
                "recovery_suggestion": "Vérifier les logs système"
            }}

    # Configuration des nœuds
    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics", False),
        ("recruiter", recruiter, "recruiter", True),
        ("rh", rh, "rh", True),
        ("talent", talent, "talent", False),
        ("onboarding", onboarding, "onboarding", False),
        ("payroll", payroll, "payroll", False)
    ]
    
    for name, agent, key, needs_data in nodes_config:
        graph.add_node(name, safe_node_wrapper(agent, key, needs_data))
    
    # Ajout des nœuds spéciaux
    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    # Configuration du workflow
    graph.set_entry_point("dataanalyst")
    
    # Branchement principal
    main_nodes = ["rh", "recruiter", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)
        graph.add_edge(node, "critique")
    
    # Flux final
    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)

    return graph
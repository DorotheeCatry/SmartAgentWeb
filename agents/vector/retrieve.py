# === FICHIER: agents/vector/retriever.py ===
import re
import os
import json
import psycopg2
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from core.llm_providers import LLMManager, LangChainLLMWrapper
from langgraph.graph import StateGraph, END
from agents.nodes.hr_agents.critique_rh_agent import CritiqueRHAgent
from agents.nodes.hr_agents.validation_rh_agent import ValidationRHAgent
from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent
from agents.nodes.hr_agents.final_rh_agent import FinalRHAgent

CHROMA_PATH = "indexes/northwind_chroma"
description_path = "indexes/northwind_schema_description.txt"
AGENT_JSON_PATH = "indexes/rh_agents.json"

def describe_pg_schema(limit_tables: int = None):
    conn = psycopg2.connect(host="localhost", port=55432, dbname="northwind", user="postgres", password="postgres")
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    results = cursor.fetchall()
    tables = [t[0] for t in results[:limit_tables]] if limit_tables else [t[0] for t in results]

    summary = []
    for table in tables:
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';")
        columns = [c[0] for c in cursor.fetchall()]
        summary.append(f"Table {table}: colonnes = {', '.join(columns)}")
    cursor.close()
    conn.close()
    return "\n".join(summary)

def suggest_agents_from_schema():
    schema_text = describe_pg_schema()
    llm = LLMManager().get_llm()

    prompt = f"""
Tu es un architecte RH intelligent.

Voici la structure de base de données :

{schema_text}

Déduis quels agents RH seraient utiles. Réponds STRICTEMENT sous forme JSON comme ci-dessous, sans explication ni phrase en dehors du JSON.

[
  {{"role": "recruiter", "description": "Responsable du recrutement"}},
  {{"role": "payroll", "description": "Gère les salaires et la paie"}}
]
"""
    result = llm.invoke(prompt)

    print("\n========== 🔎 Réponse brute LLM ==========\n" + result + "\n==========================================")

    try:
        agent_defs = json.loads(result)
        with open(AGENT_JSON_PATH, "w") as f:
            json.dump(agent_defs, f, indent=2)
        return agent_defs
    except json.JSONDecodeError:
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', result, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            agent_defs = json.loads(json_text)
            with open(AGENT_JSON_PATH, "w") as f:
                json.dump(agent_defs, f, indent=2)
            print("[✔] JSON extrait avec succès par regex.")
            return agent_defs
    print("[❌] Impossible de parser la réponse du LLM.")
    return []

def extract_pg_structure():
    conn = psycopg2.connect(host="localhost", port=55432, dbname="northwind", user="postgres", password="postgres")
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = [t[0] for t in cursor.fetchall()]

    docs = []
    for table in tables:
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';")
        columns = [c[0] for c in cursor.fetchall()]
        cursor.execute(f"SELECT * FROM {table} LIMIT 100;")
        rows = cursor.fetchall()
        for row in rows:
            text = ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
            doc = Document(page_content=f"Table {table}: {text}")
            docs.append(doc)
    cursor.close()
    conn.close()
    return docs

def ensure_index():
    if not os.path.exists(CHROMA_PATH):
        print("####### Extraction des documents #######")
        docs = extract_pg_structure()
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_PATH)
        vectorstore.persist()
        with open(description_path, "w") as f:
            f.write(describe_pg_schema())
        suggest_agents_from_schema()

def get_rh_retriever():
    ensure_index()
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings).as_retriever()

def create_dynamic_rh_graph():
    graph = StateGraph(dict)
    raw_llm = LLMManager().get_llm()
    llm = LangChainLLMWrapper(raw_llm)
    retriever = get_rh_retriever()

    if not os.path.exists(AGENT_JSON_PATH):
        agents = suggest_agents_from_schema()
    else:
        with open(AGENT_JSON_PATH, "r") as f:
            agents = json.load(f)

    def make_node(role_name, description, retriever, llm):
        def node(state: dict) -> dict:
            query = state.get("query")
            docs = retriever.invoke(query)  # nouvelle méthode recommandée
            context = "\n".join(str(doc.page_content) for doc in docs)
            prompt = f"""
            En tant qu'agent RH {role_name}, réponds à la question suivante à l'aide du contexte fourni.
            Retourne la réponse ainsi qu'un score de confiance entre 0 et 1 (float).

            Question: {query}
            Contexte:
            {context}

            Réponds sous le format JSON suivant strictement:
            {{"answer": "ta réponse", "confidence": 0.8}}
            """
            response = llm.invoke(prompt)
            try:
                parsed = json.loads(response)
                parsed["answer"] = str(parsed.get("answer", ""))  # sécurité
            except Exception:
                parsed = {"answer": str(response), "confidence": 0.0}
            state[role_name] = parsed
            return state
        return node

    for agent_def in agents:
        role = agent_def.get("role")
        description = agent_def.get("description", "")
        graph.add_node(role, make_node(role, description, retriever, llm))

    critique_agent = CritiqueRHAgent()
    validation_agent = ValidationRHAgent()
    final_agent = FinalRHAgent()

    def critique_node(state: dict):
        all_answers = [str(v["answer"]) for v in state.values() if isinstance(v, dict) and "answer" in v]
        content = "\n\n".join(all_answers)
        result = critique_agent.invoke({"content": content})
        state["critiques"] = result.get("critique", "")
        return state

    def validation_node(state: dict):
        content = state.get("critiques", "")
        result = validation_agent.invoke({"content": content})
        state["validations"] = result.get("validation", "")
        return state

    def final_node(state: dict):
        all_answers = [str(v["answer"]) for v in state.values() if isinstance(v, dict) and "answer" in v]
        answers = "\n".join(all_answers)
        critiques = state.get("critiques", "")
        validations = state.get("validations", "")
        print("[DEBUG] Réponses agents:", all_answers)
        print("[DEBUG] Critiques:", critiques)
        print("[DEBUG] Validations:", validations)
        result = final_agent.invoke({"answers": answers, "critiques": critiques, "validations": validations})
        state["final_answer"] = result.get("final_answer", "Pas de réponse.")
        return state

    graph.add_node("critique", critique_node)
    graph.add_node("validate", validation_node)
    graph.add_node("final", final_node)

    if agents:
        roles = [agent["role"] for agent in agents]
        graph.set_entry_point(roles[0])
        for i in range(len(roles) - 1):
            graph.add_edge(roles[i], roles[i+1])
        graph.add_edge(roles[-1], "critique")
        graph.add_edge("critique", "validate")
        graph.add_edge("validate", "final")
        graph.add_edge("final", END)
    else:
        graph.set_entry_point("final")
        graph.add_edge("final", END)

    return graph
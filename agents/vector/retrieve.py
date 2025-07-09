# === FICHIER: agents/vector/retriever.py ===
import re
import os, sys
import json
import psycopg2
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from core.llm_providers import LLMManager, LangChainLLMWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate

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

    print("\n========== 🔎 Réponse brute LLM ==========")
    print(result)
    print("==========================================")

    # Étape 1 : Essai normal
    try:
        agent_defs = json.loads(result)
        with open(AGENT_JSON_PATH, "w") as f:
            json.dump(agent_defs, f, indent=2)
        return agent_defs
    except json.JSONDecodeError as e:
        print(f"[!] JSON brut invalide : {e}")

    # Étape 2 : Extraction JSON via regex
    try:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            agent_defs = json.loads(json_text)
            with open(AGENT_JSON_PATH, "w") as f:
                json.dump(agent_defs, f, indent=2)
            print("[✔] JSON extrait avec succès par regex.")
            return agent_defs
        else:
            print("[!] Aucun bloc JSON détecté.")
    except Exception as e:
        print("[!] Erreur pendant parsing regex JSON:", e)

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
        print("#######Extract#######")
        docs = extract_pg_structure()
        print(f"docs: nombre = {len(docs)}, taille sys.getsizeof = {sys.getsizeof(docs)} bytes")
        
        print("#######embedding#######")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        print(f"embeddings: type = {type(embeddings)}, taille sys.getsizeof = {sys.getsizeof(embeddings)} bytes")
        
        print("#######vectorstore#######")
        #docs = docs[:3]
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_PATH)
        print(f"vectorstore: type = {type(vectorstore)}, taille sys.getsizeof = {sys.getsizeof(vectorstore)} bytes")
        
        vectorstore.persist()
        with open(description_path, "w") as f:
            f.write(describe_pg_schema())
        print("#######suggest_agents_from_schema#######")
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

        def run_retriever(query: str) -> str:
            docs = retriever.get_relevant_documents(query)
            return "\n".join(doc.page_content for doc in docs)

        def node(state: dict) -> dict:
            query = state.get("query")
            docs = retriever.get_relevant_documents(query)
            context = "\n".join(doc.page_content for doc in docs)

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
            except Exception:
                parsed = {"answer": response, "confidence": 0.0}

            state[role_name] = parsed
            return state

        return node

    # Ajout des agents dynamiquement
    for agent_def in agents:
        role = agent_def.get("role")
        description = agent_def.get("description", "")
        graph.add_node(role, make_node(role, description, retriever, llm))

    # Ajout du noeud de validation finale
    def final_decision_node(state):
        CONFIDENCE_THRESHOLD = 0.8

        results = []

        print("\n🧠 Réponses des agents :")
        for agent, response in state.items():
            if isinstance(response, dict) and "confidence" in response:
                try:
                    confidence = float(response.get("confidence", 0))
                except (ValueError, TypeError):
                    confidence = 0.0

                answer = response.get("answer", "")
                print(f" - [{agent}] (confiance {confidence:.2f}) ➤ {answer}")
                results.append((agent, confidence, answer))

        if not results:
            print("❌ Aucun agent n'a répondu.")
            state["final_answer"] = "Aucune réponse"
            return state

        # Sélection automatique si une réponse dépasse le seuil
        best_agent, best_conf, best_answer = max(results, key=lambda x: x[1])
        if best_conf >= CONFIDENCE_THRESHOLD:
            print(f"\n✅ Réponse sélectionnée automatiquement par [{best_agent}] (confiance {best_conf:.2f})")
            print(f"Réponse : {best_answer}")
            state["final_answer"] = best_answer
        else:
            print("\n⚠️ Confiance insuffisante. Aucune réponse ne dépasse le seuil.")
            user_input = input("Entrez votre réponse manuelle : ")
            state["final_answer"] = user_input

        return state

    graph.add_node("validate", final_decision_node)

    # Connexion des noeuds
    if agents:
        roles = [agent["role"] for agent in agents]
        graph.set_entry_point(roles[0])
        for i in range(len(roles) - 1):
            graph.add_edge(roles[i], roles[i+1])
        graph.add_edge(roles[-1], "validate")
        graph.add_edge("validate", END)
    else:
        graph.set_entry_point("validate")
        graph.add_edge("validate", END)

    return graph
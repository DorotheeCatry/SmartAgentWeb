import sys
from agents.graphs.smart_graph import create_smart_graph
from agents.vector.retrieve import create_dynamic_rh_graph, ensure_index
from langgraph.graph import END

def run_smart_agent():
    print("🧠 Lancement du SmartWebAgent...")
    graph = create_smart_graph().compile()
    while True:
        query = input("\nPose ta question (ou 'exit') > ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"query": query})
        print("\n=== ✅ Réponse Smart Agent ===")
        print(result.get("answer", result))
        print("=================================\n")

def run_rh_agent():
    print("👔 Préparation du Retriever RH...")
    ensure_index()
    graph = create_dynamic_rh_graph().compile()
    while True:
        query = input("\nPose ta question RH (ou 'exit') > ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"query": query})

        print("\n=== ✅ Réponse Finale de l'Agent RH ===")
        print(result.get("final_answer", "Aucune réponse générée."))
        print("=======================================\n")

if __name__ == "__main__":
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "rh":
        run_rh_agent()
    else:
        run_smart_agent()

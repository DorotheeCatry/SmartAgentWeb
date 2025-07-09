from agents.vector.retrieve import ensure_index, suggest_agents_from_schema

if __name__ == "__main__":
    #ensure_index()
    print("Index créé avec succès !")
    print("#######suggest_agents_from_schema#######")
    agents = suggest_agents_from_schema()
    print(f"Agents générés : {agents}")

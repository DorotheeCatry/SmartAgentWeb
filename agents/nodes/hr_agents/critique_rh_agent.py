from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from core.llm_providers import LLMManager

CRITIQUE_PROMPT = PromptTemplate.from_template("""
Tu es un agent expert RH en analyse critique. Voici un contenu :
{content}

Fais une critique concise, objective et constructive.
""")

llm = LLMManager().get_llm()

def critique(content: str) -> str:
    prompt_text = CRITIQUE_PROMPT.format_prompt(content=content).to_string()
    return llm.invoke(prompt_text)

class CritiqueRHAgent(Runnable):
    def invoke(self, input: dict) -> dict:
        result = critique(input["content"])
        return {"critique": result}
    
    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)

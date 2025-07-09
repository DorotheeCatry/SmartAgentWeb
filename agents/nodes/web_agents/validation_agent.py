from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from core.llm_providers import LLMManager

VALIDATION_PROMPT = PromptTemplate.from_template("""
Tu es un agent qui vérifie la validité et la fiabilité d'un contenu.
Voici le contenu :
{content}

Retourne "Valide" ou "Invalide" avec une courte justification.
""")

llm = LLMManager().get_llm()

def validate(content: str) -> str:
    prompt_text = VALIDATION_PROMPT.format(content=content)
    return llm.invoke(prompt_text)

class ValidationAgent(Runnable):
    def invoke(self, input: dict) -> dict:
        result = validate(input["content"])
        return {"validation": result}
    
    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)

from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from core.llm_providers import LLMManager

VALIDATION_PROMPT = PromptTemplate.from_template("""
Tu es un agent de validation RH. Voici une réponse :

{content}

Évalue si cette réponse est correcte, complète et fiable.
Réponds uniquement par : OUI ou NON, suivi d'une justification.
""")

llm = LLMManager().get_llm()

def validate(content: str) -> str:
    prompt_text = VALIDATION_PROMPT.format_prompt(content=content).to_string()
    return llm.invoke(prompt_text)

class ValidationRHAgent(Runnable):
    def invoke(self, input: dict) -> dict:
        result = validate(input["content"])
        return {"validation": result}
    
    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)

from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from core.llm_providers import LLMManager

FUSION_PROMPT = PromptTemplate.from_template("""
Tu es un assistant RH expert.

Voici les réponses proposées par plusieurs agents :
{answers}

Voici les critiques :
{critiques}

Voici les validations :
{validations}

Fais une synthèse claire, fiable et sans redondance. Ta réponse finale doit être concise.
""")

llm = LLMManager().get_llm()

def fuse(answers: str, critiques: str, validations: str) -> str:
    prompt_text = FUSION_PROMPT.format_prompt(
        answers=answers,
        critiques=critiques,
        validations=validations
    ).to_string()
    return llm.invoke(prompt_text)

class FinalRHAgent(Runnable):
    def invoke(self, input: dict) -> dict:
        result = fuse(
            answers=input.get("answers", ""),
            critiques=input.get("critiques", ""),
            validations=input.get("validations", "")
        )
        return {"final_answer": result}
    
    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)

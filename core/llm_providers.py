from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import AIMessage

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain.llms import Ollama as OllamaLLM

from utils.config import get_config, reset_config


class LLMProvider(ABC):
    """Interface abstraite pour les providers LLM"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2:latest", temperature: float = 0.7, **kwargs):
        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature, **kwargs)

    def invoke(self, prompt: str) -> str:
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Ollama",
            "model": self.model_name,
            "temperature": self.temperature
        }


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.llm = ChatGroq(model_name=model, api_key=api_key, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, AIMessage):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Groq",
            "model": self.model_name,
            "temperature": self.temperature
        }


class LLMManager:
    def __init__(self):
        reset_config()
        self.config = get_config()
        if self.config.llm_provider == "groq":
            self.provider = GroqProvider(
                api_key=self.config.groq_api_key,
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
        else:
            self.provider = OllamaProvider(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )

    def get_llm(self) -> LLMProvider:
        return self.provider

    def set_provider(self, provider: LLMProvider):
        self.provider = provider

    def invoke(self, prompt: str) -> str:
        return self.provider.invoke(prompt)

    def get_model_info(self) -> Dict[str, Any]:
        return self.provider.get_model_info()

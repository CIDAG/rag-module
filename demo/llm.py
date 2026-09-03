from dataclasses import dataclass
from langchain.chat_models import init_chat_model

@dataclass
class ModelMeta:
    model: str
    provider: str

def init_llm():
    model_data = ModelMeta(
        model="tinyllama",
        provider="ollama"
    )

    llm = init_chat_model(
        model=model_data.model,
        model_provider=model_data.provider
    )
    
    return llm
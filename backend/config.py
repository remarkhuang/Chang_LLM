from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    LM_STUDIO = "lm_studio"
    GROQ = "groq"
    OPENAI_COMPATIBLE = "openai_compatible"


class Settings(BaseSettings):
    app_name: str = "Free LLM Gateway"
    debug: bool = True
    
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    
    groq_api_key: Optional[str] = None
    
    default_provider: LLMProvider = LLMProvider.OLLAMA
    default_model: str = "llama2"
    
    max_tokens: int = 2048
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

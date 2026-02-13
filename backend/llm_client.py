from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass
import httpx
import json
from config import settings, LLMProvider


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[ChatMessage], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        pass


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str = None):
        super().__init__(base_url or settings.ollama_base_url)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
        return LLMResponse(
            content=data["message"]["content"],
            model=model,
            provider="ollama",
            tokens_used=data.get("eval_count", None),
            finish_reason="stop"
        )
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
    
    async def list_models(self) -> List[str]:
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            except Exception:
                return []
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
        except Exception:
            return False


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, base_url: str, api_key: str = None, provider_name: str = "openai_compatible"):
        super().__init__(base_url)
        self.api_key = api_key
        self.provider_name = provider_name
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
        
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=model,
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            finish_reason=choice.get("finish_reason")
        )
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload, headers=self._get_headers()) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
    
    async def list_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            except Exception:
                return []
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._get_headers())
                return response.status_code == 200
        except Exception:
            return False


class VLLMClient(OpenAICompatibleClient):
    def __init__(self, base_url: str = None):
        super().__init__(
            base_url or settings.vllm_base_url,
            provider_name="vllm"
        )


class LMStudioClient(OpenAICompatibleClient):
    def __init__(self, base_url: str = None):
        super().__init__(
            base_url or settings.lm_studio_base_url,
            provider_name="lm_studio"
        )


class GroqClient(OpenAICompatibleClient):
    def __init__(self, api_key: str = None):
        super().__init__(
            settings.groq_base_url,
            api_key or settings.groq_api_key,
            provider_name="groq"
        )


def get_client(provider: LLMProvider) -> BaseLLMClient:
    clients = {
        LLMProvider.OLLAMA: OllamaClient,
        LLMProvider.VLLM: VLLMClient,
        LLMProvider.LM_STUDIO: LMStudioClient,
        LLMProvider.GROQ: GroqClient,
    }
    
    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"Unknown provider: {provider}")
    
    return client_class()

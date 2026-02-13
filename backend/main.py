from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import json

from config import settings, LLMProvider
from llm_client import get_client, ChatMessage, BaseLLMClient

app = FastAPI(
    title=settings.app_name,
    description="统一接口调用多种免费LLM后端",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[MessageModel]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class ProviderInfo(BaseModel):
    name: str
    available: bool
    models: List[str]
    description: str


PROVIDER_DESCRIPTIONS = {
    "ollama": "本地运行的开源LLM服务，完全免费，支持多种模型",
    "vllm": "高性能LLM推理引擎，需要GPU支持，OpenAI兼容API",
    "lm_studio": "图形化本地LLM运行工具，易于使用，OpenAI兼容API",
    "groq": "云端LLM服务，提供免费额度，超快推理速度"
}


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "default_provider": settings.default_provider.value,
        "default_model": settings.default_model
    }


@app.get("/providers", response_model=List[ProviderInfo])
async def list_providers():
    providers = []
    for provider in LLMProvider:
        try:
            client = get_client(provider)
            available = await client.is_available()
            models = await client.list_models() if available else []
            providers.append(ProviderInfo(
                name=provider.value,
                available=available,
                models=models,
                description=PROVIDER_DESCRIPTIONS.get(provider.value, "")
            ))
        except Exception as e:
            providers.append(ProviderInfo(
                name=provider.value,
                available=False,
                models=[],
                description=PROVIDER_DESCRIPTIONS.get(provider.value, "")
            ))
    return providers


@app.get("/providers/{provider_name}/models")
async def list_models(provider_name: str):
    try:
        provider = LLMProvider(provider_name)
        client = get_client(provider)
        models = await client.list_models()
        return {"provider": provider_name, "models": models}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    provider_str = request.provider or settings.default_provider.value
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_str}")
    
    client = get_client(provider)
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail=f"Provider {provider_str} is not available")
    
    messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
    model = request.model or settings.default_model
    temperature = request.temperature or settings.temperature
    max_tokens = request.max_tokens or settings.max_tokens
    
    try:
        response = await client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return ChatResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    provider_str = request.provider or settings.default_provider.value
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_str}")
    
    client = get_client(provider)
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail=f"Provider {provider_str} is not available")
    
    messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
    model = request.model or settings.default_model
    temperature = request.temperature or settings.temperature
    max_tokens = request.max_tokens or settings.max_tokens
    
    async def generate():
        try:
            async for chunk in client.chat_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class ProxyRequest(BaseModel):
    url: str
    method: str = "POST"
    headers: dict = {}
    body: dict = {}


@app.post("/proxy")
async def proxy_request(request: ProxyRequest):
    import httpx
    
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if request.method.upper() == "GET":
                response = await client.get(request.url, headers=headers)
            else:
                response = await client.post(request.url, headers=headers, json=request.body)
        
        return {
            "status_code": response.status_code,
            "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

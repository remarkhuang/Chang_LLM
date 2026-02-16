from fastapi import FastAPI, HTTPException, Header, Depends
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

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if settings.gateway_api_key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token format")
        
        token = authorization.split(" ")[1]
        if token != settings.gateway_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return authorization


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
    failover: Optional[bool] = False
    failover_order: Optional[List[str]] = None
    api_keys: Optional[List[str]] = None



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
async def chat(request: ChatRequest, _ = Depends(verify_api_key)):
    providers_to_try = []
    if request.failover:
        order = request.failover_order or settings.failover_order
        # 確保順序中包含當前請求的 provider (如果有的話且不在順序中)
        if request.provider and request.provider not in order:
            providers_to_try.append(request.provider)
        providers_to_try.extend(order)
        # 去重但保持順序
        providers_to_try = list(dict.fromkeys(providers_to_try))
    else:
        providers_to_try = [request.provider or settings.default_provider.value]
    
    last_error = None
    for provider_str in providers_to_try:
        try:
            provider = LLMProvider(provider_str)
            
            # 獲取該 Provider 可用的 API Keys
            # 如果目前嘗試的是 request.provider，則使用傳入的 api_keys
            keys_to_try = [""] # 預設至少嘗試一次 (針對不需要 Key 的本地模型)
            if provider_str == request.provider and request.api_keys:
                keys_to_try = [k for k in request.api_keys if k.strip()]
            
            # 如果沒有傳入 Key 或是其他故障轉移後的 Provider，則嘗試從設定檔獲取單一 Key
            if not keys_to_try:
                if provider_str == "groq" and settings.groq_api_key:
                    keys_to_try = [settings.groq_api_key]
                else:
                    keys_to_try = [""]

            for api_key in keys_to_try:
                try:
                    # 動態建立帶有特定 API Key 的 client
                    # 這裡稍微重構 get_client 的概念
                    if provider == LLMProvider.GROQ:
                        client = GroqClient(api_key=api_key)
                    elif provider == LLMProvider.OLLAMA:
                        client = OllamaClient()
                    elif provider == LLMProvider.VLLM:
                        client = VLLMClient()
                    elif provider == LLMProvider.LM_STUDIO:
                        client = LMStudioClient()
                    else:
                        client = get_client(provider)
                    
                    if not await client.is_available():
                        continue
                        
                    messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
                    model = request.model or settings.default_model
                    
                    response = await client.chat(
                        messages=messages,
                        model=model,
                        temperature=request.temperature or settings.temperature,
                        max_tokens=request.max_tokens or settings.max_tokens
                    )
                    return ChatResponse(
                        content=response.content,
                        model=response.model,
                        provider=response.provider,
                        tokens_used=response.tokens_used,
                        finish_reason=response.finish_reason
                    )
                except Exception as key_e:
                    last_error = key_e
                    print(f"Key for {provider_str} failed: {str(key_e)}")
                    continue # 嘗試下一個 Key
                    
        except Exception as e:
            last_error = e
            print(f"Provider {provider_str} failed: {str(e)}")
            continue
            
    raise HTTPException(status_code=500, detail=f"All attempts failed. Last error: {str(last_error)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, _ = Depends(verify_api_key)):
    providers_to_try = []
    if request.failover:
        order = request.failover_order or settings.failover_order
        if request.provider and request.provider not in order:
            providers_to_try.append(request.provider)
        providers_to_try.extend(order)
        providers_to_try = list(dict.fromkeys(providers_to_try))
    else:
        providers_to_try = [request.provider or settings.default_provider.value]
    
    async def generate():
        last_error = None
        for provider_str in providers_to_try:
            try:
                provider = LLMProvider(provider_str)
                
                # 獲取 API Keys
                keys_to_try = [""]
                if provider_str == request.provider and request.api_keys:
                    keys_to_try = [k for k in request.api_keys if k.strip()]
                
                if not keys_to_try:
                    if provider_str == "groq" and settings.groq_api_key:
                        keys_to_try = [settings.groq_api_key]
                    else:
                        keys_to_try = [""]

                for api_key in keys_to_try:
                    try:
                        # 動態建立 client
                        if provider == LLMProvider.GROQ:
                            client = GroqClient(api_key=api_key)
                        elif provider == LLMProvider.OLLAMA:
                            client = OllamaClient()
                        elif provider == LLMProvider.VLLM:
                            client = VLLMClient()
                        elif provider == LLMProvider.LM_STUDIO:
                            client = LMStudioClient()
                        else:
                            client = get_client(provider)
                        
                        if not await client.is_available():
                            continue
                        
                        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
                        model = request.model or settings.default_model
                        
                        has_yielded = False
                        try:
                            async for chunk in client.chat_stream(
                                messages=messages,
                                model=model,
                                temperature=request.temperature or settings.temperature,
                                max_tokens=request.max_tokens or settings.max_tokens
                            ):
                                yield f"data: {json.dumps({'content': chunk, 'provider': provider_str})}\n\n"
                                has_yielded = True
                            
                            if has_yielded:
                                yield "data: [DONE]\n\n"
                                return
                        except Exception as inner_e:
                            if has_yielded:
                                yield f"data: {json.dumps({'error': f'Stream interrupted: {str(inner_e)}'})}\n\n"
                                return
                            else:
                                raise inner_e # 尚未輸出，嘗試下一個 Key 或 Provider
                    except Exception as key_e:
                        last_error = key_e
                        print(f"Key for {provider_str} failed in stream: {str(key_e)}")
                        continue
                        
            except Exception as e:
                last_error = e
                print(f"Provider {provider_str} failed in stream: {str(e)}")
                continue
        
        yield f"data: {json.dumps({'error': f'All attempts failed. Last error: {str(last_error)}'})}\n\n"
    
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
async def proxy_request(request: ProxyRequest, _ = Depends(verify_api_key)):
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

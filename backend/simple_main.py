from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx

app = FastAPI(title="Free LLM Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProxyRequest(BaseModel):
    url: str
    method: str = "POST"
    headers: Dict[str, str] = {}
    body: Dict[str, Any] = {}


@app.get("/")
async def root():
    return {"name": "Free LLM Proxy", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/proxy")
async def proxy(request: ProxyRequest):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']}
    
    print(f"[Proxy] URL: {request.url}")
    print(f"[Proxy] Headers: {headers}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if request.method.upper() == "GET":
                response = await client.get(request.url, headers=headers)
            else:
                response = await client.post(request.url, headers=headers, json=request.body)
        
        print(f"[Proxy] Response status: {response.status_code}")
        
        try:
            body = response.json()
        except:
            body = response.text
        
        return {
            "status_code": response.status_code,
            "body": body
        }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        print(f"[Proxy] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

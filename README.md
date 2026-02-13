# Free LLM Gateway

统一接口调用多种免费LLM后端的项目，支持 Ollama、vLLM、LM Studio 和 Groq。

## 项目结构

```
free_LLM/
├── backend/
│   ├── main.py          # FastAPI 主应用
│   ├── config.py        # 配置管理
│   ├── llm_client.py    # LLM 客户端实现
│   ├── requirements.txt # Python 依赖
│   └── .env.example     # 环境变量示例
├── frontend/
│   └── index.html       # Web 前端界面
├── start.bat            # Windows 启动脚本
├── start.sh             # Linux/Mac 启动脚本
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp backend/.env.example backend/.env
# 编辑 .env 文件配置你的 API keys
```

### 3. 启动服务

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

或手动启动：
```bash
# 后端
cd backend
python -m uvicorn main:app --reload --port 8000

# 前端
cd frontend
python -m http.server 3000
```

### 4. 访问应用

- 前端界面: http://localhost:3000
- API 文档: http://localhost:8000/docs

---

## LLM 提供商配置详解

### 🦙 Ollama (推荐本地使用)

**特点:**
- 完全免费，本地运行
- 支持多种开源模型
- 隐私安全，数据不出本地
- 安装简单，开箱即用

**安装:**
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 访问 https://ollama.com/download 下载安装
```

**使用:**
```bash
# 下载并运行模型
ollama run llama2
ollama run mistral
ollama run codellama

# 查看已安装模型
ollama list

# API 调用
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

**Python 调用:**
```python
import httpx

response = httpx.post("http://localhost:11434/api/chat", json={
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False
})
print(response.json()["message"]["content"])
```

---

### ⚡ vLLM (高性能推理)

**特点:**
- 高吞吐量，适合生产环境
- 需要 GPU 支持
- OpenAI 兼容 API
- 支持 PagedAttention 优化

**安装:**
```bash
pip install vllm
```

**启动服务:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

**Python 调用:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

### 🖥️ LM Studio (图形界面)

**特点:**
- 图形界面，易于使用
- 支持从 HuggingFace 下载模型
- 自动提供 OpenAI 兼容 API
- 适合非技术用户

**安装:**
1. 访问 https://lmstudio.ai 下载
2. 安装后打开应用
3. 搜索并下载模型 (如 Llama 2, Mistral)
4. 启动本地服务器 (默认端口 1234)

**API 调用:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### 🚀 Groq (云端免费)

**特点:**
- 云端服务，无需本地资源
- 超快推理速度 (LPU 芯片)
- 有免费额度
- 支持大模型 (Llama 2 70B, Mixtral)

**注册:**
1. 访问 https://console.groq.com
2. 注册账号获取 API Key

**配置:**
```bash
export GROQ_API_KEY=your_api_key
```

**API 调用:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="your_api_key"
)

response = client.chat.completions.create(
    model="llama2-70b-4096",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**可用模型:**
- `llama2-70b-4096` - Llama 2 70B
- `mixtral-8x7b-32768` - Mixtral 8x7B
- `gemma-7b-it` - Gemma 7B

---

## API 接口说明

### 获取提供商列表
```
GET /providers
```

### 获取模型列表
```
GET /providers/{provider}/models
```

### 发送聊天请求
```
POST /chat
{
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": "ollama",
    "model": "llama2",
    "temperature": 0.7,
    "max_tokens": 2048
}
```

### 流式聊天
```
POST /chat/stream
```
返回 Server-Sent Events (SSE) 格式的流式响应。

---

## 常见问题

### Q: Ollama 连接失败？
确保 Ollama 服务正在运行：
```bash
ollama serve
```

### Q: vLLM 启动失败？
确保有足够的 GPU 显存，或尝试量化模型：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-GPTQ \
    --quantization gptq
```

### Q: 如何添加新的 LLM 后端？
在 `llm_client.py` 中添加新的客户端类，继承 `BaseLLMClient` 并实现相应方法。

---

## 许可证

MIT License

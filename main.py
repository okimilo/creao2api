from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import json
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

app = FastAPI(title="CREAO Web2API - 完整上下文版")

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
COOKIE = os.getenv("COOKIE")
if not BEARER_TOKEN or not COOKIE:
    raise Exception("请在 .env 中填写 BEARER_TOKEN 和 COOKIE")

BASE_URL = "https://agent.creao.ai/api/agent/run"

HEADERS = { ... }  # 保持你之前的那一整段 HEADERS 不变

client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    openai_data = await request.json()
    messages = openai_data.get("messages", [])
    stream = openai_data.get("stream", True)
    model = openai_data.get("model", "google/gemini-3.1-pro-preview")

    # ==================== 暴力拼接完整上下文 ====================
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
            
        if role == "system":
            prompt += f"[System Details/设定信息]:\n{content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
    
    # 最后引导模型继续以 Assistant 身份回复
    prompt += "Assistant: "
    
    if not prompt.strip():
        raise HTTPException(400, "No valid content found in messages")
    # ============================================================

    payload = {
        "prompt": prompt,
        "mode": "copilot",
        "chatModelId": model,
        "skillIds": [],
        "displayContent": prompt[:200]  # displayContent 只给前端预览用，截短一点
    }

    async def generate():
        async with client.stream("POST", BASE_URL, json=payload, headers=HEADERS, cookies=COOKIE) as resp:
            if resp.status_code != 200:
                error_text = await resp.aread()
                yield f'data: {{"error": "CREAO 返回 {resp.status_code}"}}\n\n'
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    msg_type = data.get("type")

                    if msg_type == "text_delta":
                        content = data.get("content", "")
                        if content:
                            chunk = {
                                "id": "chatcmpl-creao",
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    elif msg_type == "done":
                        final_chunk = {
                            "id": "chatcmpl-creao",
                            "object": "chat.completion.chunk",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                except:
                    continue

    if stream:
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        raise HTTPException(400, "暂不支持非流式")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import json
import os
import asyncio

app = FastAPI(title="CREAO Web2API - Railway 版")

# ==================== 从 Railway 环境变量读取 ====================
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
COOKIE = os.getenv("COOKIE")

if not BEARER_TOKEN or not COOKIE:
    raise Exception("缺少环境变量：请在 Railway 设置 BEARER_TOKEN 和 COOKIE")

BASE_URL = "https://agent.creao.ai/api/agent/run"

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "authorization": f"Bearer {BEARER_TOKEN}",
    "content-type": "application/json",
    "origin": "https://agent.creao.ai",
    "referer": "https://agent.creao.ai/chat",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    openai_data = await request.json()
    messages = openai_data.get("messages", [])
    stream = openai_data.get("stream", True)
    model = openai_data.get("model", "google/gemini-3.1-pro-preview")

    # 暴力拼接完整上下文（角色设定 + 全部历史记录）
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
    prompt += "Assistant: "

    if not prompt.strip():
        raise HTTPException(400, "No valid content found in messages")

    payload = {
        "prompt": prompt,
        "mode": "copilot",
        "chatModelId": model,
        "skillIds": [],
        "displayContent": prompt[:200]
    }

    async def generate():
        async with client.stream("POST", BASE_URL, json=payload, headers=HEADERS, cookies=COOKIE) as resp:
            if resp.status_code != 200:
                yield f'data: {{"error": "CREAO 返回 {resp.status_code}"}}\n\n'
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "text_delta":
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
                    elif data.get("type") == "done":
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

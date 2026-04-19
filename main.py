from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import os
import asyncio
from collections import deque
import http.cookies

app = FastAPI(title="CREAO Web2API - 完整内容纯转发版")

ACCOUNTS_JSON = os.getenv("ACCOUNTS")
if not ACCOUNTS_JSON:
    raise Exception("缺少环境变量 ACCOUNTS")
ACCOUNTS = json.loads(ACCOUNTS_JSON)
print(f"✅ 已加载 {len(ACCOUNTS)} 个 CREAO 账号")

account_queue = deque(ACCOUNTS)

BASE_URL = "https://agent.creao.ai/api/agent/run"
client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)

def parse_cookies(cookie_str: str) -> dict:
    cookie = http.cookies.SimpleCookie()
    cookie.load(cookie_str)
    return {k: v.value for k, v in cookie.items()}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    openai_data = await request.json()
    messages = openai_data.get("messages", [])
    stream = openai_data.get("stream", True)
    model = openai_data.get("model", "google/gemini-3.1-pro-preview")

    # 轮询账号
    account = account_queue.popleft()
    account_queue.append(account)

    BEARER_TOKEN = account["bearer"]
    COOKIE_STR = account["cookie"]
    COOKIES_DICT = parse_cookies(COOKIE_STR)

    # ==================== 完整把 Chatbox 发来的 messages 转成一个大 prompt ====================
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            if role == "system":
                prompt += f"[System]\n{content}\n\n"
            elif role == "user":
                prompt += f"[User]\n{content}\n\n"
            elif role == "assistant":
                prompt += f"[Assistant]\n{content}\n\n"
    if not prompt.strip():
        raise HTTPException(400, "No content found in messages")

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

    payload = {
        "prompt": prompt,           # 完整内容
        "mode": "copilot",
        "chatModelId": model,
        "skillIds": [],
        "displayContent": prompt[:200]
    }

    collected_content = ""

    async def generate():
        nonlocal collected_content
        async with client.stream("POST", BASE_URL, json=payload, headers=HEADERS, cookies=COOKIES_DICT) as resp:
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
                            collected_content += content
                            if stream:
                                chunk = {
                                    "id": "chatcmpl-creao",
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                    elif data.get("type") == "done":
                        if stream:
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
        async for _ in generate():
            pass
        full_response = {
            "id": "chatcmpl-creao",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": collected_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return JSONResponse(full_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

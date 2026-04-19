from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import json
import os
import asyncio
from collections import deque
import http.cookies  # 新增：用于解析 cookie 字符串

app = FastAPI(title="CREAO Web2API - 多账户轮询版")

# ==================== 多账户配置 ====================
ACCOUNTS_JSON = os.getenv("ACCOUNTS")
if not ACCOUNTS_JSON:
    raise Exception("缺少环境变量 ACCOUNTS")

try:
    ACCOUNTS = json.loads(ACCOUNTS_JSON)
except:
    raise Exception("ACCOUNTS 格式错误，必须是合法 JSON 数组")

if not isinstance(ACCOUNTS, list) or len(ACCOUNTS) == 0:
    raise Exception("ACCOUNTS 至少需要 1 个账号")

print(f"✅ 已加载 {len(ACCOUNTS)} 个 CREAO 账号")

account_queue = deque(ACCOUNTS)

BASE_URL = "https://agent.creao.ai/api/agent/run"

client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)

# ==================== 新增：解析 cookie 字符串为 dict ====================
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

    # 轮询取出当前账号
    account = account_queue.popleft()
    account_queue.append(account)

    BEARER_TOKEN = account["bearer"]
    COOKIE_STR = account["cookie"]          # 原始字符串
    COOKIES_DICT = parse_cookies(COOKIE_STR)  # 转成 dict

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

    # 暴力拼接上下文
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
        async with client.stream(
            "POST", 
            BASE_URL, 
            json=payload, 
            headers=HEADERS, 
            cookies=COOKIES_DICT   # ← 这里改成 dict
        ) as resp:
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

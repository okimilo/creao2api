from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import os
import asyncio
from collections import deque
import http.cookies
import uuid


app = FastAPI(title="CREAO Web2API - C测试版(threadId + 最后一条user)")


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


def get_last_user_message(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(text)
                if parts:
                    return "\n".join(parts).strip()
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    openai_data = await request.json()
    messages = openai_data.get("messages", [])
    stream = openai_data.get("stream", True)
    model = openai_data.get("model", "google/gemini-3.1-pro-preview")

    if not messages:
        raise HTTPException(400, "messages 不能为空")

    # 轮询账号
    account = account_queue.popleft()
    account_queue.append(account)

    BEARER_TOKEN = account["bearer"]
    COOKIE_STR = account["cookie"]
    COOKIES_DICT = parse_cookies(COOKIE_STR)

    # ==================== C版核心：只取最后一条 user ====================
    prompt = get_last_user_message(messages)
    if not prompt:
        raise HTTPException(400, "未找到最后一条 user message")

    # ==================== C版核心：生成/透传 threadId ====================
    # 优先从请求里拿；没有就生成一个
    thread_id = (
        openai_data.get("threadId")
        or openai_data.get("thread_id")
        or request.headers.get("x-thread-id")
        or str(uuid.uuid4())
    )

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": f"Bearer {BEARER_TOKEN}",
        "content-type": "application/json",
        "origin": "https://agent.creao.ai",
        "referer": f"https://agent.creao.ai/chat?threadId={thread_id}",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }

    payload = {
        "prompt": prompt,
        "threadId": thread_id,
        "mode": "copilot",
        "chatModelId": model,
        "skillIds": [],
        "displayContent": prompt
    }

    collected_content = ""

    async def generate():
        nonlocal collected_content

        async with client.stream(
            "POST",
            BASE_URL,
            json=payload,
            headers=headers,
            cookies=COOKIES_DICT
        ) as resp:
            if resp.status_code != 200:
                err_text = await resp.aread()
                err_text = err_text.decode("utf-8", errors="ignore")
                yield f'data: {json.dumps({"error": f"CREAO 返回 {resp.status_code}", "detail": err_text[:500]})}\n\n'
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception:
                    continue

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
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                elif data.get("type") == "done":
                    if stream:
                        final_chunk = {
                            "id": "chatcmpl-creao",
                            "object": "chat.completion.chunk",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    break

    if stream:
        response = StreamingResponse(generate(), media_type="text/event-stream")
        response.headers["X-CREAO-Thread-Id"] = thread_id
        return response
    else:
        async for _ in generate():
            pass

        full_response = {
            "id": "chatcmpl-creao",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": collected_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "threadId": thread_id
        }
        response = JSONResponse(full_response)
        response.headers["X-CREAO-Thread-Id"] = thread_id
        return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

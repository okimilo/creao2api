from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import os
import asyncio
from collections import deque
import http.cookies
from typing import Any, Optional


app = FastAPI(title="CREAO Web2API - C Test Version")


ACCOUNTS_JSON = os.getenv("ACCOUNTS")
if not ACCOUNTS_JSON:
    raise Exception("缺少环境变量 ACCOUNTS")

try:
    ACCOUNTS = json.loads(ACCOUNTS_JSON)
except Exception as e:
    raise Exception(f"ACCOUNTS 解析失败: {e}")

if not isinstance(ACCOUNTS, list) or not ACCOUNTS:
    raise Exception("ACCOUNTS 必须是非空数组")

print(f"✅ 已加载 {len(ACCOUNTS)} 个 CREAO 账号")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-3.1-pro-preview")
DEFAULT_THREAD_ID = os.getenv("DEFAULT_THREAD_ID", "").strip()

# 可选：调试日志
DEBUG_LOG = os.getenv("DEBUG_LOG", "1") == "1"

account_queue = deque(ACCOUNTS)

BASE_URL = "https://agent.creao.ai/api/agent/run"

client = httpx.AsyncClient(
    timeout=180.0,
    follow_redirects=True
)


def log(*args):
    if DEBUG_LOG:
        print(*args)


def parse_cookies(cookie_str: str) -> dict:
    cookie = http.cookies.SimpleCookie()
    cookie.load(cookie_str)
    return {k: v.value for k, v in cookie.items()}


def get_last_user_message(messages: list) -> str:
    """
    只取最后一条 user 消息。
    兼容：
    - {"content": "文本"}
    - {"content": [{"type": "text", "text": "..."}]}
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")

        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
            text = "\n".join(parts).strip()
            if text:
                return text

    return ""


def get_thread_id(openai_data: dict, request: Request) -> str:
    """
    优先级：
    1. JSON: threadId
    2. JSON: thread_id
    3. Header: x-thread-id
    4. 环境变量 DEFAULT_THREAD_ID
    """
    candidates = [
        openai_data.get("threadId"),
        openai_data.get("thread_id"),
        request.headers.get("x-thread-id"),
        DEFAULT_THREAD_ID,
    ]

    for item in candidates:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            return value

    return ""


def make_openai_error(message: str, err_type: str = "upstream_error", code: int = 500):
    return JSONResponse(
        status_code=code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code
            }
        }
    )


def current_ts() -> int:
    try:
        return int(asyncio.get_event_loop().time())
    except Exception:
        import time
        return int(time.time())


def make_sse_error_chunk(message: str, code: int = 500, err_type: str = "upstream_error") -> str:
    payload = {
        "id": "chatcmpl-creao",
        "object": "error",
        "created": current_ts(),
        "error": {
            "message": message,
            "type": err_type,
            "code": code
        }
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "CREAO Web2API",
        "mode": "C test version",
        "default_model": DEFAULT_MODEL,
        "has_default_thread_id": bool(DEFAULT_THREAD_ID)
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        openai_data = await request.json()
    except Exception:
        raise HTTPException(400, "请求体不是合法 JSON")

    messages = openai_data.get("messages", [])
    stream = openai_data.get("stream", True)
    model = openai_data.get("model", DEFAULT_MODEL)

    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages 不能为空且必须是数组")

    # 轮询账号
    account = account_queue.popleft()
    account_queue.append(account)

    bearer_token = account.get("bearer", "").strip()
    cookie_str = account.get("cookie", "").strip()

    if not bearer_token or not cookie_str:
        return make_openai_error("账号配置缺少 bearer 或 cookie", "config_error", 500)

    cookies_dict = parse_cookies(cookie_str)

    # C版核心：只取最后一条 user
    prompt = get_last_user_message(messages)
    if not prompt:
        raise HTTPException(400, "未找到最后一条 user message")

    # C版核心：使用真实 threadId
    thread_id = get_thread_id(openai_data, request)

    referer = "https://agent.creao.ai/chat"
    if thread_id:
        referer = f"https://agent.creao.ai/chat?threadId={thread_id}"

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": f"Bearer {bearer_token}",
        "content-type": "application/json",
        "origin": "https://agent.creao.ai",
        "referer": referer,
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }

    payload = {
        "prompt": prompt,
        "mode": "copilot",
        "chatModelId": model,
        "skillIds": [],
        "displayContent": prompt
    }

    if thread_id:
        payload["threadId"] = thread_id

    log("===== CREAO REQUEST =====")
    log("model:", model)
    log("thread_id:", thread_id or "(empty)")
    log("referer:", referer)
    log("prompt_preview:", prompt[:200].replace("\n", "\\n"))
    log("=========================")

    collected_content = ""

    async def generate():
        nonlocal collected_content

        try:
            async with client.stream(
                "POST",
                BASE_URL,
                json=payload,
                headers=headers,
                cookies=cookies_dict
            ) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    err_text = raw.decode("utf-8", errors="ignore")
                    msg = f"CREAO 返回 {resp.status_code}"
                    if err_text:
                        msg += f": {err_text[:500]}"

                    log("❌ Upstream error:", msg)

                    if stream:
                        yield make_sse_error_chunk(msg, resp.status_code, "upstream_error")
                        yield "data: [DONE]\n\n"
                    else:
                        raise RuntimeError(msg)
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except Exception:
                        # 有些行不是 JSON，直接跳过
                        continue

                    event_type = data.get("type")

                    if event_type == "text_delta":
                        content = data.get("content", "")
                        if content:
                            collected_content += content
                            if stream:
                                chunk = {
                                    "id": "chatcmpl-creao",
                                    "object": "chat.completion.chunk",
                                    "created": current_ts(),
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

                    elif event_type == "done":
                        if stream:
                            final_chunk = {
                                "id": "chatcmpl-creao",
                                "object": "chat.completion.chunk",
                                "created": current_ts(),
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

                    elif event_type in ("error", "failed"):
                        msg = data.get("message") or json.dumps(data, ensure_ascii=False)
                        if stream:
                            yield make_sse_error_chunk(msg, 502, "upstream_error")
                            yield "data: [DONE]\n\n"
                        else:
                            raise RuntimeError(msg)
                        return

                    else:
                        # 其他事件例如 thinking_delta 暂时忽略
                        continue

        except RuntimeError:
            raise
        except Exception as e:
            log("❌ Server error:", repr(e))
            if stream:
                yield make_sse_error_chunk(str(e), 500, "server_error")
                yield "data: [DONE]\n\n"
            else:
                raise

    if stream:
        response = StreamingResponse(generate(), media_type="text/event-stream")
        if thread_id:
            response.headers["X-CREAO-Thread-Id"] = thread_id
        return response

    try:
        async for _ in generate():
            pass
    except RuntimeError as e:
        return make_openai_error(str(e), "upstream_error", 502)
    except Exception as e:
        return make_openai_error(str(e), "server_error", 500)

    full_response = {
        "id": "chatcmpl-creao",
        "object": "chat.completion",
        "created": current_ts(),
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
        }
    }

    if thread_id:
        full_response["threadId"] = thread_id

    response = JSONResponse(full_response)
    if thread_id:
        response.headers["X-CREAO-Thread-Id"] = thread_id
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

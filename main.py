"""
Lark Bot - FastAPI + Claude AI
Chức năng: Tạo form (Bitable), tạo task trong Lark qua chat
"""

import os
import json
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from anthropic import Anthropic

app = FastAPI(title="Lark AI Bot")
claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ─── Lark Config ─────────────────────────────────────────────────────────────
LARK_APP_ID     = os.environ["LARK_APP_ID"]
LARK_APP_SECRET = os.environ["LARK_APP_SECRET"]
LARK_BASE_URL   = "https://open.larksuite.com/open-apis"

# ─── Token cache ─────────────────────────────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}


def safe_json(resp) -> dict:
    """Parse JSON an toàn, không crash nếu response rỗng."""
    try:
        return resp.json()
    except Exception:
        return {"error": f"HTTP {resp.status_code}", "body": resp.text[:200]}


async def get_access_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LARK_BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        )
        data = resp.json()
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expires_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


# ─── Lark API helpers ─────────────────────────────────────────────────────────

async def send_message(chat_id: str, text: str):
    """Gửi text message vào chat/DM."""
    token = await get_access_token()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{LARK_BASE_URL}/im/v1/messages?receive_id_type=open_id",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            },
        )
    return safe_json(r)


async def create_task(summary: str, due_date: str = None, description: str = "") -> dict:
    token = await get_access_token()
    payload: dict = {
        "summary": summary,
        "description": description,
        "origin": {"platform_i18n_name": {"en_us": "AI Bot"}},
    }
    if due_date:
        payload["due"] = {"timestamp": str(due_date)}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{LARK_BASE_URL}/task/v2/tasks",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return safe_json(r)


async def create_bitable_table(app_token: str, table_name: str, fields: list) -> dict:
    token = await get_access_token()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{LARK_BASE_URL}/bitable/v1/apps/{app_token}/tables",
            headers={"Authorization": f"Bearer {token}"},
            json={"table": {"name": table_name, "fields": fields}},
        )
    return safe_json(r)


async def add_bitable_record(app_token: str, table_id: str, fields: dict) -> dict:
    token = await get_access_token()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{LARK_BASE_URL}/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            headers={"Authorization": f"Bearer {token}"},
            json={"fields": fields},
        )
    return safe_json(r)


# ─── Claude tool definitions ──────────────────────────────────────────────────

TOOLS = [
    {
        "name": "create_task",
        "description": "Tạo task mới trong Lark Tasks",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Tên/tiêu đề task"},
                "description": {"type": "string", "description": "Mô tả chi tiết"},
                "due_date": {"type": "string", "description": "Unix timestamp deadline (ms), optional"},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "create_bitable_table",
        "description": "Tạo bảng/form mới trong Lark Base (Bitable)",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string", "description": "Token của Lark Base app"},
                "table_name": {"type": "string", "description": "Tên bảng"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "type": {"type": "integer", "description": "1=text,2=number,3=select,5=date"},
                        },
                    },
                },
            },
            "required": ["app_token", "table_name", "fields"],
        },
    },
    {
        "name": "add_bitable_record",
        "description": "Thêm record vào bảng Bitable",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string"},
                "table_id": {"type": "string"},
                "fields": {"type": "object"},
            },
            "required": ["app_token", "table_id", "fields"],
        },
    },
    {
        "name": "reply_message",
        "description": "Trả lời tin nhắn cho user",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]

SYSTEM_PROMPT = """Bạn là AI assistant trong Lark, hỗ trợ quản lý công việc.
Bạn có thể tạo task, tạo bảng Bitable, thêm record.
Khi user yêu cầu tạo task → dùng tool create_task ngay.
Khi user hỏi bình thường → dùng tool reply_message để trả lời.
Trả lời tiếng Việt, ngắn gọn, thân thiện."""


async def execute_tool(tool_name: str, tool_input: dict, chat_id: str) -> str:
    try:
        if tool_name == "create_task":
            result = await create_task(
                summary=tool_input["summary"],
                due_date=tool_input.get("due_date"),
                description=tool_input.get("description", ""),
            )
            if result.get("error"):
                return f"Lỗi tạo task: {result}"
            task_id = result.get("data", {}).get("task", {}).get("id", "")
            return f"✅ Task '{tool_input['summary']}' đã tạo! ID: {task_id}"

        elif tool_name == "create_bitable_table":
            result = await create_bitable_table(
                app_token=tool_input["app_token"],
                table_name=tool_input["table_name"],
                fields=tool_input["fields"],
            )
            table_id = result.get("data", {}).get("table_id", "")
            return f"✅ Bảng '{tool_input['table_name']}' đã tạo! ID: {table_id}"

        elif tool_name == "add_bitable_record":
            await add_bitable_record(
                app_token=tool_input["app_token"],
                table_id=tool_input["table_id"],
                fields=tool_input["fields"],
            )
            return "✅ Đã thêm record!"

        elif tool_name == "reply_message":
            await send_message(chat_id, tool_input["text"])
            return "sent"

    except Exception as e:
        return f"Lỗi: {str(e)}"

    return f"Tool không hỗ trợ: {tool_name}"


# ─── Claude agentic loop ──────────────────────────────────────────────────────

async def process_with_claude(user_message: str, chat_id: str):
    try:
        messages = [{"role": "user", "content": user_message}]

        for _ in range(5):
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        await send_message(chat_id, block.text)
                return

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await execute_tool(block.name, block.input, chat_id)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if not tool_results:
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        # Luôn gửi phản hồi dù có lỗi
        try:
            await send_message(chat_id, f"Xin lỗi, có lỗi xảy ra: {str(e)[:100]}")
        except Exception:
            pass


# ─── Webhook ──────────────────────────────────────────────────────────────────

# Chống duplicate events
_processed_events: set = set()


@app.post("/webhook/lark")
async def lark_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"code": 0})

    # URL verification
    if body.get("type") == "url_verification":
        return JSONResponse({"challenge": body.get("challenge")})

    # Chống duplicate
    event_id = body.get("header", {}).get("event_id", "")
    if event_id and event_id in _processed_events:
        return JSONResponse({"code": 0})
    if event_id:
        _processed_events.add(event_id)
        if len(_processed_events) > 1000:
            _processed_events.clear()

    # Lấy event
    header = body.get("header", {})
    event_type = header.get("event_type", "") or body.get("event", {}).get("type", "")
    event = body.get("event", {})

    if event_type == "im.message.receive_v1":
        sender = event.get("sender", {})
        if sender.get("sender_type") == "app":
            return JSONResponse({"code": 0})

        msg = event.get("message", {})
        chat_id = msg.get("chat_id", "")

        # Lấy open_id để reply DM
        open_id = sender.get("sender_id", {}).get("open_id", "")
        reply_to = chat_id if chat_id else open_id

        content_raw = msg.get("content", "{}")
        try:
            text = json.loads(content_raw).get("text", "").strip()
        except Exception:
            text = content_raw

        if text and reply_to:
            import asyncio
            asyncio.create_task(process_with_claude(text, reply_to))

    return JSONResponse({"code": 0})


@app.get("/health")
async def health():
    return {"status": "ok"}

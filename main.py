"""
Lark Bot - FastAPI + Claude AI
Chức năng: Tạo form (Bitable), tạo task trong Lark qua chat
"""

import os
import json
import hmac
import hashlib
import time
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from anthropic import Anthropic

app = FastAPI(title="Lark AI Bot")
claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ─── Lark Config ─────────────────────────────────────────────────────────────
LARK_APP_ID         = os.environ["LARK_APP_ID"]
LARK_APP_SECRET     = os.environ["LARK_APP_SECRET"]
LARK_ENCRYPT_KEY    = os.environ.get("LARK_ENCRYPT_KEY", "")
LARK_VERIFY_TOKEN   = os.environ.get("LARK_VERIFY_TOKEN", "")
LARK_BASE_URL       = "https://open.larksuite.com/open-apis"

# ─── Token cache ─────────────────────────────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}


async def get_access_token() -> str:
    """Lấy tenant access token, cache lại để tránh gọi thừa."""
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
        await client.post(
            f"{LARK_BASE_URL}/im/v1/messages?receive_id_type=chat_id",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            },
        )


async def create_task(summary: str, due_date: str = None, description: str = "") -> dict:
    """Tạo task mới trong Lark Tasks."""
    token = await get_access_token()
    payload = {
        "summary": summary,
        "description": description,
        "due": {"timestamp": due_date} if due_date else {},
        "origin": {
            "platform_i18n_name": {"zh_cn": "AI Bot", "en_us": "AI Bot"},
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LARK_BASE_URL}/task/v1/tasks",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return resp.json()


async def create_bitable_table(app_token: str, table_name: str, fields: list[dict]) -> dict:
    """Tạo bảng mới trong Bitable (Lark Base / form)."""
    token = await get_access_token()
    payload = {
        "table": {
            "name": table_name,
            "fields": fields,
        }
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LARK_BASE_URL}/bitable/v1/apps/{app_token}/tables",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return resp.json()


async def add_bitable_record(app_token: str, table_id: str, fields: dict) -> dict:
    """Thêm 1 record vào Bitable table."""
    token = await get_access_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LARK_BASE_URL}/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            headers={"Authorization": f"Bearer {token}"},
            json={"fields": fields},
        )
    return resp.json()


# ─── Claude tool definitions ──────────────────────────────────────────────────

TOOLS = [
    {
        "name": "create_task",
        "description": "Tạo task mới trong Lark Tasks",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Tên/tiêu đề task"},
                "description": {"type": "string", "description": "Mô tả chi tiết task"},
                "due_date": {"type": "string", "description": "Deadline dạng Unix timestamp (string), optional"},
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
                "table_name": {"type": "string", "description": "Tên bảng/form"},
                "fields": {
                    "type": "array",
                    "description": "Danh sách field của bảng",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "type": {"type": "integer", "description": "1=text, 2=number, 3=single_select, 4=multi_select, 5=date"},
                        },
                    },
                },
            },
            "required": ["app_token", "table_name", "fields"],
        },
    },
    {
        "name": "add_bitable_record",
        "description": "Thêm dữ liệu/record vào bảng Bitable",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string"},
                "table_id": {"type": "string"},
                "fields": {"type": "object", "description": "Dict key-value các field cần điền"},
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
                "text": {"type": "string", "description": "Nội dung trả lời"},
            },
            "required": ["text"],
        },
    },
]

SYSTEM_PROMPT = """Bạn là AI assistant tích hợp vào Lark, giúp user quản lý công việc.
Bạn có thể:
1. Tạo task trong Lark Tasks
2. Tạo bảng/form trong Lark Base (Bitable)
3. Thêm record vào bảng Bitable

Khi user yêu cầu tạo task/form, hãy dùng tool tương ứng ngay lập tức.
Luôn xác nhận lại với user sau khi hoàn thành.
Trả lời bằng tiếng Việt, ngắn gọn, thân thiện."""


# ─── Tool executor ────────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, tool_input: dict, chat_id: str) -> str:
    if tool_name == "create_task":
        result = await create_task(
            summary=tool_input["summary"],
            due_date=tool_input.get("due_date"),
            description=tool_input.get("description", ""),
        )
        task_id = result.get("data", {}).get("task", {}).get("id", "unknown")
        return f"Task đã tạo thành công! ID: {task_id}"

    elif tool_name == "create_bitable_table":
        result = await create_bitable_table(
            app_token=tool_input["app_token"],
            table_name=tool_input["table_name"],
            fields=tool_input["fields"],
        )
        table_id = result.get("data", {}).get("table_id", "unknown")
        return f"Bảng '{tool_input['table_name']}' đã tạo! Table ID: {table_id}"

    elif tool_name == "add_bitable_record":
        result = await add_bitable_record(
            app_token=tool_input["app_token"],
            table_id=tool_input["table_id"],
            fields=tool_input["fields"],
        )
        return "Record đã thêm thành công!"

    elif tool_name == "reply_message":
        await send_message(chat_id, tool_input["text"])
        return "Đã gửi tin nhắn"

    return f"Tool {tool_name} không được hỗ trợ"


# ─── Claude agentic loop ──────────────────────────────────────────────────────

async def process_with_claude(user_message: str, chat_id: str):
    """Xử lý tin nhắn qua Claude với tool use."""
    messages = [{"role": "user", "content": user_message}]

    for _ in range(5):  # max 5 vòng agentic loop
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Nếu không còn tool call → gửi text cuối cùng
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    await send_message(chat_id, block.text)
            return

        # Xử lý tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = await execute_tool(block.name, block.input, chat_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Thêm vào messages để tiếp tục loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ─── Webhook endpoint ─────────────────────────────────────────────────────────

@app.post("/webhook/lark")
async def lark_webhook(request: Request):
    body = await request.json()

    # 1. URL Verification challenge (Lark gọi khi setup webhook)
    if body.get("type") == "url_verification":
        return JSONResponse({"challenge": body.get("challenge")})

    # 2. Event callback
    event = body.get("event", {})
    if not event:
        return JSONResponse({"code": 0})

    event_type = event.get("type") or body.get("header", {}).get("event_type", "")

    # Chỉ xử lý tin nhắn text
    if event_type in ("im.message.receive_v1", "message"):
        msg = event.get("message", {})
        sender = event.get("sender", {})

        # Bỏ qua tin nhắn từ bot chính nó
        if sender.get("sender_type") == "app":
            return JSONResponse({"code": 0})

        chat_id = msg.get("chat_id", "")
        content = msg.get("content", "{}")
        try:
            text = json.loads(content).get("text", "").strip()
        except Exception:
            text = content

        if text and chat_id:
            import asyncio
            asyncio.create_task(process_with_claude(text, chat_id))

    return JSONResponse({"code": 0})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Lark AI Bot"}

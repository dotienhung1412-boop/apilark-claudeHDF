"""
Lark Bot - FastAPI + Claude AI
Linh hoạt, tiếng Việt, hỗ trợ: tạo form/bảng, tạo task, giao việc cho team
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

# ─── Config ───────────────────────────────────────────────────────────────────
LARK_APP_ID     = os.environ["LARK_APP_ID"]
LARK_APP_SECRET = os.environ["LARK_APP_SECRET"]
LARK_BASE_URL   = "https://open.larksuite.com/open-apis"

# Danh sách thành viên team: {"Tên": "open_id"}
# Điền vào Railway Variables: TEAM_MEMBERS={"Hưng":"ou_xxx","Nam":"ou_yyy"}
try:
    TEAM_MEMBERS: dict = json.loads(os.environ.get("TEAM_MEMBERS", "{}"))
except Exception:
    TEAM_MEMBERS = {}

# ─── Token cache ──────────────────────────────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}

# ─── Deduplication (event_id + message_id) ────────────────────────────────────
_seen_event_ids: set = set()
_seen_message_ids: set = set()
MAX_CACHE = 500


def is_duplicate(event_id: str, message_id: str) -> bool:
    """Trả về True nếu event/message đã xử lý rồi."""
    if event_id and event_id in _seen_event_ids:
        return True
    if message_id and message_id in _seen_message_ids:
        return True
    # Lưu lại
    if event_id:
        _seen_event_ids.add(event_id)
        if len(_seen_event_ids) > MAX_CACHE:
            _seen_event_ids.clear()
    if message_id:
        _seen_message_ids.add(message_id)
        if len(_seen_message_ids) > MAX_CACHE:
            _seen_message_ids.clear()
    return False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_json(resp) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"error": f"HTTP {resp.status_code}", "raw": resp.text[:300]}


async def get_access_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{LARK_BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        )
        data = r.json()
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expires_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


async def lark_post(path: str, payload: dict) -> dict:
    token = await get_access_token()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{LARK_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return safe_json(r)


async def lark_get(path: str, params: dict = None) -> dict:
    token = await get_access_token()
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{LARK_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
        )
    return safe_json(r)


async def send_message(receive_id: str, text: str, id_type: str = "chat_id"):
    return await lark_post(
        f"/im/v1/messages?receive_id_type={id_type}",
        {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}),
        },
    )


# ─── Tools definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "tao_task",
        "description": "Tạo task mới trong Lark Tasks và có thể giao cho thành viên cụ thể",
        "input_schema": {
            "type": "object",
            "properties": {
                "ten_task": {"type": "string", "description": "Tiêu đề task"},
                "mo_ta": {"type": "string", "description": "Mô tả chi tiết (optional)"},
                "deadline_timestamp": {"type": "string", "description": "Unix timestamp ms cho deadline (optional)"},
                "giao_cho": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tên thành viên cần giao task (ví dụ: ['Hưng', 'Nam']). Bot tự tra open_id.",
                },
            },
            "required": ["ten_task"],
        },
    },
    {
        "name": "tao_bang_bitable",
        "description": "Tạo bảng/form mới trong Lark Base (Bitable) với cấu trúc tuỳ ý",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "Token của Lark Base. Lấy từ URL khi mở Lark Base: phần sau /base/ và trước dấu ?",
                },
                "ten_bang": {"type": "string", "description": "Tên bảng cần tạo"},
                "fields": {
                    "type": "array",
                    "description": "Danh sách cột. type: 1=văn bản, 2=số, 3=đơn lựa chọn, 4=đa lựa chọn, 5=ngày, 11=checkbox, 13=số điện thoại, 15=URL",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "type": {"type": "integer"},
                        },
                        "required": ["field_name", "type"],
                    },
                },
            },
            "required": ["app_token", "ten_bang", "fields"],
        },
    },
    {
        "name": "them_record_bitable",
        "description": "Thêm dữ liệu mới vào bảng Bitable",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string"},
                "table_id": {"type": "string"},
                "du_lieu": {
                    "type": "object",
                    "description": "Dict key=tên cột, value=giá trị cần điền",
                },
            },
            "required": ["app_token", "table_id", "du_lieu"],
        },
    },
    {
        "name": "xem_danh_sach_team",
        "description": "Xem danh sách thành viên team đã được cấu hình",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "tra_loi",
        "description": "Gửi tin nhắn trả lời cho user",
        "input_schema": {
            "type": "object",
            "properties": {
                "noi_dung": {"type": "string", "description": "Nội dung trả lời"},
            },
            "required": ["noi_dung"],
        },
    },
]

# ─── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    team_str = (
        "\n".join([f"- {name}" for name in TEAM_MEMBERS.keys()])
        if TEAM_MEMBERS
        else "Chưa có (cần cấu hình TEAM_MEMBERS)"
    )
    return f"""Bạn là AI assistant của Hưng, tích hợp vào Lark để hỗ trợ công việc digital marketing và quản lý team.

Bạn có thể làm những việc sau trong Lark:
1. **Tạo task** và giao cho thành viên cụ thể
2. **Tạo bảng/form** trong Lark Base với cấu trúc bất kỳ theo yêu cầu
3. **Thêm dữ liệu** vào bảng có sẵn
4. **Trả lời câu hỏi** về công việc, ads, marketing

**Thành viên team hiện tại:**
{team_str}

**Nguyên tắc làm việc:**
- Luôn trả lời bằng tiếng Việt
- Khi user yêu cầu tạo task/form → dùng tool ngay, không hỏi lại nhiều
- Khi tạo bảng Bitable → tự suy luận cấu trúc cột phù hợp từ mô tả của user
- Nếu cần app_token Bitable mà user chưa cung cấp → hỏi user lấy từ URL Lark Base
- Nếu giao task mà tên người không có trong danh sách team → báo user cập nhật TEAM_MEMBERS
- Xác nhận ngắn gọn sau khi hoàn thành (không dài dòng)
- Nếu không chắc yêu cầu → hỏi lại 1 câu ngắn gọn

**Ví dụ mapping yêu cầu → hành động:**
- "tạo task X giao cho Nam" → tool tao_task với giao_cho=["Nam"]
- "tạo form theo dõi kế hoạch ads" → tool tao_bang_bitable với các cột phù hợp
- "thêm dữ liệu vào bảng X" → tool them_record_bitable
- "team có ai?" → tool xem_danh_sach_team"""


# ─── Tool executor ────────────────────────────────────────────────────────────

async def execute_tool(name: str, args: dict, chat_id: str, id_type: str) -> str:
    try:
        if name == "tao_task":
            # Resolve tên → open_id
            assignee_ids = []
            unresolved = []
            for ten in args.get("giao_cho", []):
                uid = TEAM_MEMBERS.get(ten)
                if uid:
                    assignee_ids.append(uid)
                else:
                    unresolved.append(ten)

            payload: dict = {
                "summary": args["ten_task"],
                "description": args.get("mo_ta", ""),
                "origin": {"platform_i18n_name": {"en_us": "Lark AI Bot"}},
            }
            if args.get("deadline_timestamp"):
                payload["due"] = {"timestamp": str(args["deadline_timestamp"])}

            result = await lark_post("/task/v2/tasks", payload)
            task_id = result.get("data", {}).get("task", {}).get("id", "")

            msg = f"✅ Task **{args['ten_task']}** đã tạo!"
            if assignee_ids:
                msg += f"\n👤 Giao cho: {', '.join([k for k,v in TEAM_MEMBERS.items() if v in assignee_ids])}"
            if unresolved:
                msg += f"\n⚠️ Không tìm thấy: {', '.join(unresolved)} — cần cập nhật TEAM_MEMBERS"
            if task_id:
                msg += f"\n🔗 Task ID: {task_id}"
            return msg

        elif name == "tao_bang_bitable":
            result = await lark_post(
                f"/bitable/v1/apps/{args['app_token']}/tables",
                {"table": {"name": args["ten_bang"], "fields": args["fields"]}},
            )
            if result.get("code", 0) != 0:
                return f"❌ Lỗi tạo bảng: {result.get('msg', result)}"
            table_id = result.get("data", {}).get("table_id", "")
            return f"✅ Bảng **{args['ten_bang']}** đã tạo! Table ID: `{table_id}`"

        elif name == "them_record_bitable":
            result = await lark_post(
                f"/bitable/v1/apps/{args['app_token']}/tables/{args['table_id']}/records",
                {"fields": args["du_lieu"]},
            )
            if result.get("code", 0) != 0:
                return f"❌ Lỗi thêm record: {result.get('msg', result)}"
            return "✅ Đã thêm dữ liệu vào bảng!"

        elif name == "xem_danh_sach_team":
            if not TEAM_MEMBERS:
                return "⚠️ Chưa có thành viên nào. Thêm vào Railway Variables: TEAM_MEMBERS={\"Tên\":\"open_id\"}"
            lines = "\n".join([f"- {k}: `{v}`" for k, v in TEAM_MEMBERS.items()])
            return f"👥 **Danh sách team:**\n{lines}"

        elif name == "tra_loi":
            await send_message(chat_id, args["noi_dung"], id_type)
            return "sent"

    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

    return f"Tool không hỗ trợ: {name}"


# ─── Claude agentic loop ──────────────────────────────────────────────────────

async def process_with_claude(user_message: str, chat_id: str, id_type: str = "chat_id"):
    try:
        messages = [{"role": "user", "content": user_message}]
        system = build_system_prompt()

        for _ in range(6):
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # End turn → gửi text reply
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        await send_message(chat_id, block.text, id_type)
                return

            # Xử lý tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await execute_tool(block.name, block.input, chat_id, id_type)
                    # Với tool tra_loi, tin nhắn đã được gửi trực tiếp
                    if block.name != "tra_loi" and result != "sent":
                        await send_message(chat_id, result, id_type)
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
        try:
            await send_message(chat_id, f"Xin lỗi, có lỗi: {str(e)[:150]}", id_type)
        except Exception:
            pass


# ─── Webhook ──────────────────────────────────────────────────────────────────

@app.post("/webhook/lark")
async def lark_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"code": 0})

    # URL verification
    if body.get("type") == "url_verification":
        return JSONResponse({"challenge": body.get("challenge")})

    header = body.get("header", {})
    event_type = header.get("event_type", "") or body.get("event", {}).get("type", "")
    event = body.get("event", {})

    if event_type == "im.message.receive_v1":
        msg = event.get("message", {})
        sender = event.get("sender", {})

        # Bỏ qua tin nhắn từ bot
        if sender.get("sender_type") == "app":
            return JSONResponse({"code": 0})

        event_id = header.get("event_id", "")
        message_id = msg.get("message_id", "")

        # Chống duplicate bằng cả event_id lẫn message_id
        if is_duplicate(event_id, message_id):
            return JSONResponse({"code": 0})

        # Xác định reply target
        chat_type = msg.get("chat_type", "")
        chat_id = msg.get("chat_id", "")
        open_id = sender.get("sender_id", {}).get("open_id", "")

        if chat_type == "p2p" and open_id:
            reply_to = open_id
            id_type = "open_id"
        else:
            reply_to = chat_id
            id_type = "chat_id"

        # Parse text
        content_raw = msg.get("content", "{}")
        try:
            text = json.loads(content_raw).get("text", "").strip()
        except Exception:
            text = content_raw.strip()

        if text and reply_to:
            import asyncio
            asyncio.create_task(process_with_claude(text, reply_to, id_type))

    return JSONResponse({"code": 0})


@app.get("/health")
async def health():
    return {"status": "ok", "team_members": list(TEAM_MEMBERS.keys())}

"""
mcp_server.py — QI MCP 工具服务器（标准 MCP 协议版）

协议：Model Context Protocol (MCP) — https://spec.modelcontextprotocol.io
传输：SSE（Server-Sent Events），兼容 HTTP 部署（Zeabur / Railway / Heroku 等）
SDK：modelcontextprotocol/python-sdk  (pip install mcp)

启动方式：
  python mcp_server.py                  # SSE 模式，默认端口 8001
  MCP_PORT=9000 python mcp_server.py

Claude Desktop / 客户端配置（SSE）：
  {
    "mcpServers": {
      "qi": {
        "url": "http://localhost:8001/sse"
      }
    }
  }

环境变量：
  SUPABASE_URL / SUPABASE_KEY
  GMAIL_TOKEN   (JSON 字符串)
  MCP_PORT      (默认 8001)
"""

from __future__ import annotations

import os, json, base64 as b64
from datetime import datetime, timezone, timedelta
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import Response
import uvicorn

load_dotenv()

TZ8 = timezone(timedelta(hours=8))


def now8() -> str:
    return datetime.now(TZ8).replace(tzinfo=None).isoformat(timespec="seconds")


# ── Supabase ───────────────────────────────────────────────────────────────

_sb = None


def get_sb():
    global _sb
    if _sb is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if url and key:
            from supabase import create_client
            _sb = create_client(url, key)
    return _sb


# ── MCP Server 实例 ────────────────────────────────────────────────────────

server = Server("qi-mcp")


# ── 工具列表 ───────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="control_toy",
            description="控制玩具（震动 / 吸吮 / 伸缩），数值 0 表示关闭",
            inputSchema={
                "type": "object",
                "properties": {
                    "vibrate_mode":      {"type": "integer", "default": 0},
                    "vibrate_intensity": {"type": "integer", "default": 0},
                    "suck_mode":         {"type": "integer", "default": 0},
                    "suck_intensity":    {"type": "integer", "default": 0},
                    "stretch_mode":      {"type": "integer", "default": 0},
                    "stretch_intensity": {"type": "integer", "default": 0},
                },
            },
        ),
        types.Tool(
            name="send_email",
            description="通过 Gmail 发邮件",
            inputSchema={
                "type": "object",
                "required": ["to", "subject", "body"],
                "properties": {
                    "to":      {"type": "string"},
                    "subject": {"type": "string"},
                    "body":    {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="read_emails",
            description="读取 Gmail 最新邮件",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 5}},
            },
        ),
    ]


# ── 工具调用入口 ───────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    handlers = {
        "control_toy":           _control_toy,
        "send_email":            _send_email,
        "read_emails":           lambda a: _read_emails(int(a.get("limit", 5))),
    }
    fn = handlers.get(name)
    if fn is None:
        result = f'工具 "{name}" 未找到'
    else:
        try:
            result = fn(arguments)
        except Exception as e:
            result = f"执行出错：{e}"
    return [types.TextContent(type="text", text=str(result))]


# ── 工具实现 ───────────────────────────────────────────────────────────────

def _control_toy(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    params = {k: int(args.get(k, 0)) for k in (
        "vibrate_mode", "vibrate_intensity",
        "suck_mode", "suck_intensity",
        "stretch_mode", "stretch_intensity",
    )}
    sb.table("toy_commands").insert({
        "command": "control",
        "params": json.dumps(params),
        "status": "pending",
        "created_at": now8(),
    }).execute()
    active = ", ".join(f"{k}={v}" for k, v in params.items() if v)
    return f"指令已发送：{active or '全部关闭'}"


def _get_gmail_service():
    token_json = os.getenv("GMAIL_TOKEN", "")
    if not token_json:
        raise Exception("GMAIL_TOKEN 未配置")
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    td = json.loads(token_json)
    creds = Credentials(
        token=td.get("token"),
        refresh_token=td.get("refresh_token"),
        token_uri=td.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=td.get("client_id"),
        client_secret=td.get("client_secret"),
        scopes=td.get("scopes"),
    )
    return build("gmail", "v1", credentials=creds)


def _send_email(args: dict) -> str:
    from email.mime.text import MIMEText
    to, subject, body = args.get("to"), args.get("subject"), args.get("body")
    if not (to and subject and body):
        return "缺少参数（to / subject / body）"
    service = _get_gmail_service()
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = b64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"邮件已发送至 {to}"


def _read_emails(limit: int = 5) -> str:
    service = _get_gmail_service()
    resp = service.users().messages().list(userId="me", maxResults=limit).execute()
    msgs = resp.get("messages", [])
    if not msgs:
        return "暂无邮件"
    results = []
    for m in msgs:
        detail = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        results.append(
            f"发件人：{headers.get('From','')}\n"
            f"主题：{headers.get('Subject','')}\n"
            f"时间：{headers.get('Date','')}"
        )
    return "\n---\n".join(results)


# ── iOS 推送端点（保留，不属于 MCP 协议，作为额外 HTTP 端点） ──────────────

def _parse_ios_screentime(raw: str) -> list[dict]:
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(
            r'^(.+?)\s*\((?:(\d+)\s*[時时hr]+\s*)?(?:(\d+)\s*[分min]+\s*)?(?:(\d+)\s*[秒sec]+\s*)?\)',
            line,
        )
        if not m:
            continue
        name = m.group(1).strip()
        hours = int(m.group(2) or 0)
        minutes = int(m.group(3) or 0)
        seconds = int(m.group(4) or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if name and total > 0:
            results.append({"name": name, "duration_seconds": float(total)})
    return results


async def push_screentime_endpoint(request: Request):
    from starlette.responses import JSONResponse
    data = await request.json() if request.method == "POST" else {}
    sb = get_sb()
    if not sb:
        return JSONResponse({"error": "Supabase 未配置"}, status_code=500)
    try:
        today = datetime.now(TZ8).strftime("%Y-%m-%d")
        raw = data.get("apps") or ""
        if isinstance(raw, list):
            parsed = [
                {"name": (i.get("name") or "").strip(),
                 "duration_seconds": float(i.get("duration") or 0)}
                for i in raw if i.get("name") and i.get("duration")
            ]
        else:
            parsed = _parse_ios_screentime(str(raw))
        sb.table("screentime_daily").delete().eq("date", today).execute()
        rows = [
            {"id": str(uuid.uuid4()), "date": today,
             "app_name": i["name"], "duration_seconds": i["duration_seconds"],
             "created_at": now8()}
            for i in parsed
        ]
        if rows:
            sb.table("screentime_daily").insert(rows).execute()
        return JSONResponse({"status": "ok", "apps_count": len(rows)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def push_app_event_endpoint(request: Request):
    from starlette.responses import JSONResponse
    data = {}
    if request.method == "POST":
        try:
            data = await request.json()
        except Exception:
            pass
    app_name = (request.query_params.get("app") or data.get("app") or "").strip()
    event = (request.query_params.get("event") or data.get("event") or "open").strip().lower()
    sb = get_sb()
    if not sb:
        return JSONResponse({"error": "Supabase 未配置"}, status_code=500)
    if not app_name:
        return JSONResponse({"error": "缺少 app 字段"}, status_code=400)
    if event not in ("open", "close"):
        return JSONResponse({"error": "event 必须是 open 或 close"}, status_code=400)
    try:
        sb.table("app_events").insert({
            "id": str(uuid.uuid4()),
            "app_name": app_name,
            "event": event,
            "created_at": now8(),
        }).execute()
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def health_endpoint(request: Request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "service": "qi-mcp", "protocol": "MCP/SSE"})


# ── Starlette 应用（SSE 传输） ─────────────────────────────────────────────

sse_transport = SseServerTransport("/messages/")


async def handle_sse(scope, receive, send):
    async with sse_transport.connect_sse(scope, receive, send) as streams:
        await server.run(
            streams[0], streams[1],
            server.create_initialization_options(),
        )


async def handle_messages(scope, receive, send):
    await sse_transport.handle_post_message(scope, receive, send)


starlette_app = Starlette(
    routes=[
        Route("/",                health_endpoint),
        Route("/push/screentime", push_screentime_endpoint, methods=["POST"]),
        Route("/push/app_event",  push_app_event_endpoint,  methods=["GET", "POST"]),
    ],
)


async def _asgi_inner(scope, receive, send):
    if scope["type"] == "http":
        path = scope.get("path", "")
        if path == "/sse":
            await handle_sse(scope, receive, send)
            return
        if path.startswith("/messages"):
            await handle_messages(scope, receive, send)
            return
    await starlette_app(scope, receive, send)


# CORS を /sse・/messages も含む全ルートに適用
asgi_app = CORSMiddleware(
    app=_asgi_inner,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", os.getenv("PORT", "8001")))
    print(f"QI MCP Server (SSE) :{port}")
    print(f"SSE endpoint : http://0.0.0.0:{port}/sse")
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)

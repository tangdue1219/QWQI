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

import os, json, uuid, re, base64 as b64
from datetime import datetime, timezone, timedelta
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
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
            name="post_moment",
            description="在朋友圈发一条动态",
            inputSchema={
                "type": "object",
                "required": ["content"],
                "properties": {"content": {"type": "string", "description": "动态内容"}},
            },
        ),
        types.Tool(
            name="read_moments",
            description="查看朋友圈动态（含评论和 id）",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 10, "description": "返回条数"}},
            },
        ),
        types.Tool(
            name="reply_moment_comment",
            description="在动态下发评论或回复",
            inputSchema={
                "type": "object",
                "required": ["moment_id", "content"],
                "properties": {
                    "moment_id":   {"type": "string"},
                    "content":     {"type": "string"},
                    "reply_to_id": {"type": "string", "description": "被回复的评论 id（可选）"},
                },
            },
        ),
        types.Tool(
            name="list_books",
            description="查看书架（含书籍 id）",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="read_book_chapter",
            description="读书籍某章内容",
            inputSchema={
                "type": "object",
                "required": ["book_id"],
                "properties": {
                    "book_id":     {"type": "string"},
                    "chapter_num": {"type": "integer", "default": 1},
                },
            },
        ),
        types.Tool(
            name="check_coread_pending",
            description="查看渡正在读的书和章节（渡翻到新章且棲还没有批注时会出现）",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="write_book_annotation",
            description="在书籍某章某段落写棲的批注",
            inputSchema={
                "type": "object",
                "required": ["book_id", "chapter_num", "paragraph_idx", "paragraph_text", "content"],
                "properties": {
                    "book_id":        {"type": "string"},
                    "chapter_num":    {"type": "integer"},
                    "paragraph_idx":  {"type": "integer", "description": "段落索引（从 0 开始）"},
                    "paragraph_text": {"type": "string", "description": "被批注的段落原文（取前 100 字）"},
                    "content":        {"type": "string", "description": "批注内容"},
                },
            },
        ),
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
        "post_moment":           _post_moment,
        "read_moments":          _read_moments,
        "reply_moment_comment":  _reply_moment_comment,
        "list_books":            _list_books,
        "read_book_chapter":     _read_book_chapter,
        "check_coread_pending":  _check_coread_pending,
        "write_book_annotation": _write_book_annotation,
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

def _post_moment(args: dict) -> str:
    sb = get_sb()
    row = {
        "id": str(uuid.uuid4()),
        "content": args.get("content", ""),
        "author": "qi",
        "created_at": now8(),
    }
    if sb:
        sb.table("moments").insert(row).execute()
    return "动态已发布"


def _read_moments(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    limit = int(args.get("limit", 10))
    res = (
        sb.table("moments")
        .select("id,content,author,created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    moments = res.data or []
    if not moments:
        return "还没有动态。"
    results = []
    for m in moments:
        cres = (
            sb.table("moment_comments")
            .select("*")
            .eq("moment_id", m["id"])
            .order("created_at")
            .execute()
        )
        coms = cres.data or []
        com_text = ""
        if coms:
            com_text = "\n" + "\n".join(
                f"  {'棲' if c['author']=='qi' else '渡'}"
                f"{'回复@'+('棲' if c.get('reply_to_author')=='qi' else '渡') if c.get('reply_to_author') else ''}："
                f"{c['content']} [id:{c['id']}]"
                for c in coms
            )
        results.append(
            f"[{m['created_at'][:10]}] [id:{m['id']}] "
            f"{'棲' if m['author']=='qi' else '渡'}：{m['content']}{com_text}"
        )
    return "\n\n".join(results)


def _reply_moment_comment(args: dict) -> str:
    sb = get_sb()
    row = {
        "id":              str(uuid.uuid4()),
        "moment_id":       args.get("moment_id", ""),
        "content":         args.get("content", ""),
        "author":          "qi",
        "reply_to_id":     args.get("reply_to_id") or None,
        "reply_to_author": None,
        "created_at":      now8(),
    }
    if sb:
        sb.table("moment_comments").insert(row).execute()
    return "已发布评论"


def _list_books(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    res = (
        sb.table("books")
        .select("id,title,author,total_chapters,progress")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    books = res.data or []
    if not books:
        return "书架上没有书籍。"
    return "\n".join(
        f"[id:{b['id']}] 《{b['title']}》- {b['author']} 共{b['total_chapters']}章 进度{b['progress']}%"
        for b in books
    )


def _read_book_chapter(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    book_id = args.get("book_id", "")
    num = int(args.get("chapter_num", 1))
    book_res = sb.table("books").select("title,total_chapters").eq("id", book_id).single().execute()
    if not book_res.data:
        return f"找不到书籍 id={book_id}"
    book = book_res.data
    ch_res = (
        sb.table("book_chapters")
        .select("title,content")
        .eq("book_id", book_id)
        .eq("chapter_num", num)
        .single()
        .execute()
    )
    if not ch_res.data:
        return f"《{book['title']}》没有第{num}章（共{book['total_chapters']}章）"
    ch = ch_res.data
    content = ch.get("content", "")
    ch_title = ch.get("title", f"第{num}章")
    excerpt = content[:2000]
    truncated = len(content) > 2000
    return (
        f"《{book['title']}》{ch_title}（第{num}章/共{book['total_chapters']}章）\n"
        f"{excerpt}{'…（截断）' if truncated else ''}"
    )


def _check_coread_pending(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    res = (
        sb.table("book_progress")
        .select("book_id,current_chapter,updated_at")
        .eq("reader", "qi_pending")
        .execute()
    )
    rows = res.data or []
    if not rows:
        return "暂无待共读章节。"
    results = []
    for r in rows:
        book_res = sb.table("books").select("title").eq("id", r["book_id"]).single().execute()
        title = book_res.data.get("title", r["book_id"]) if book_res.data else r["book_id"]
        results.append(f"《{title}》第{r['current_chapter']}章（渡 {r['updated_at'][:16]} 翻到这里）")
    return "渡正在读：\n" + "\n".join(results)


def _write_book_annotation(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    book_id = args.get("book_id", "")
    chapter_num = int(args.get("chapter_num", 1))
    paragraph_idx = int(args.get("paragraph_idx", 0))
    paragraph_text = str(args.get("paragraph_text", ""))[:200]
    content = args.get("content", "").strip()
    if not content:
        return "TOOL_ERROR: content 不能为空"
    sb.table("book_annotations").insert({
        "book_id":        book_id,
        "chapter_num":    chapter_num,
        "paragraph_idx":  paragraph_idx,
        "paragraph_text": paragraph_text,
        "author":         "qi",
        "content":        content,
        "created_at":     now8(),
    }).execute()
    sb.table("book_progress").delete().eq("book_id", book_id).eq("reader", "qi_pending").execute()
    sb.table("book_progress").upsert(
        {"book_id": book_id, "reader": "qi", "current_chapter": chapter_num, "updated_at": now8()},
        upsert_keys=["book_id", "reader"],
    ).execute()
    return f"TOOL_OK: 批注已写入第{chapter_num}章第{paragraph_idx}段"


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
    ]
)


async def asgi_app(scope, receive, send):
    if scope["type"] == "http":
        path = scope.get("path", "")
        if path == "/sse":
            await handle_sse(scope, receive, send)
            return
        if path.startswith("/messages"):
            await handle_messages(scope, receive, send)
            return
    await starlette_app(scope, receive, send)


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", os.getenv("PORT", "8001")))
    print(f"QI MCP Server (SSE) :{port}")
    print(f"SSE endpoint : http://0.0.0.0:{port}/sse")
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)

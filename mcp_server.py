"""
mcp_server.py — QI MCP 工具服务器

端点：
  GET  /                健康检查
  GET  /tools           工具定义列表
  POST /tools/call      执行工具
  POST /push/screentime 模块一：iOS 快捷指令定时推送今日 App 汇总时长
  POST /push/app_event  模块二：iOS 自动化推送 App open/close 事件 + 电量

工具列表：
  write_diary / read_diary
  write_memory_event / store_archive_memory
  check_calendar / add_calendar_event
  post_moment / reply_moment_comment / read_moments
  add_memo
  read_messages / list_books / read_book_chapter
  log_period
  check_du_status / check_screentime / check_battery / control_toy
  send_email / read_emails

Supabase 新增表（见末尾注释）：
  screentime_daily  — 模块一每日汇总
  app_events        — 模块二 open/close 事件流
  battery_logs      — 电量上报

角色约定：raw_conversations.role 用 du/qi（兼容旧 user/assistant）
"""

from __future__ import annotations

import os, json, uuid, re
from datetime import datetime, timezone, timedelta
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

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



# ── 工具定义 ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "post_moment",
        "description": "在朋友圈发一条动态",
        "inputSchema": {
            "type": "object",
            "required": ["content"],
            "properties": {"content": {"type": "string"}},
        },
    },
    {
        "name": "list_books",
        "description": "查看书架（含书籍id）",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_book_chapter",
        "description": "读书籍某章内容",
        "inputSchema": {
            "type": "object",
            "required": ["book_id"],
            "properties": {
                "book_id":     {"type": "string"},
                "chapter_num": {"type": "integer", "default": 1},
            },
        },
    },
    {
        "name": "read_moments",
        "description": "查看朋友圈动态（含评论和id）",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        },
    },
    {
        "name": "reply_moment_comment",
        "description": "在动态下发评论或回复",
        "inputSchema": {
            "type": "object",
            "required": ["moment_id", "content"],
            "properties": {
                "moment_id":  {"type": "string"},
                "content":    {"type": "string"},
                "reply_to_id":{"type": "string"},
            },
        },
    },
    {
        "name": "control_toy",
        "description": "控制玩具（震动/吸吮/伸缩），数值 0 表示关闭",
        "inputSchema": {
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
    },
    {
        "name": "send_email",
        "description": "通过 Gmail 发邮件",
        "inputSchema": {
            "type": "object",
            "required": ["to", "subject", "body"],
            "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
        },
    },
    {
        "name": "read_emails",
        "description": "读取 Gmail 最新邮件",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 5}},
        },
    },
]

# ── 路由 ───────────────────────────────────────────────────────────────────


@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "qi-mcp"})


@app.get("/tools")
def list_tools():
    return jsonify({"tools": TOOLS})


@app.post("/tools/call")
def call_tool():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    args = data.get("arguments", data.get("args", {})) or {}
    if not name:
        return jsonify({"error": "name required"}), 400
    try:
        result = _dispatch(name, args)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ── 模块一：iOS 快捷指令定时推送今日汇总 ──────────────────────────────────

@app.post("/push/screentime")
def push_screentime():
    """
    iOS 快捷指令（定时，如每小时/每天）推送今日各 App 汇总时长。

    iOS「取得今天的 App 与网站活动」会以纯文本输出，格式如：
      Chrome (1時16分)
      WeChat (5分)
      Safari (30秒)

    快捷指令配置：
      1. 「取得今天的 App 与网站活动」
      2. 「取得 URL 内容」POST https://域名/push/screentime
         要求内文类型：JSON
         加入欄位：键 = apps，值 = 活动变量
    """
    # iOS 传来的可能是 {"apps": "Chrome (1時16分)\nWeChat (5分)\n..."} 纯文本
    data = request.get_json(force=True, silent=True) or {}
    sb = get_sb()
    if not sb:
        return jsonify({"error": "Supabase 未配置"}), 500

    def _parse_ios_screentime(raw: str) -> list[dict]:
        """
        解析 iOS 屏幕时间纯文本，返回 [{name, duration_seconds}, ...]
        支持格式：
          App名 (X時YY分)  /  App名 (YY分)  /  App名 (YY秒)  /  App名 (X時)
        """
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # 匹配括号内时长，兼容中文「時/分/秒」和英文「hr/min/sec」
            m = re.search(
                r'^(.+?)\s*\((?:(\d+)\s*[時时hr]+\s*)?(?:(\d+)\s*[分min]+\s*)?(?:(\d+)\s*[秒sec]+\s*)?\)',
                line
            )
            if not m:
                continue
            name    = m.group(1).strip()
            hours   = int(m.group(2) or 0)
            minutes = int(m.group(3) or 0)
            seconds = int(m.group(4) or 0)
            total   = hours * 3600 + minutes * 60 + seconds
            if name and total > 0:
                results.append({"name": name, "duration_seconds": float(total)})
        return results

    try:
        today = datetime.now(TZ8).strftime("%Y-%m-%d")

        # 取出原始文本，支持两种传法：
        #   {"apps": "Chrome...\nWeChat..."}   ← 快捷指令 JSON 键值对
        #   {"apps": [...]}                    ← 万一以后变成数组也兼容
        raw = data.get("apps") or data.get("application") or ""

        if isinstance(raw, list):
            # 数组格式（兼容旧逻辑）
            parsed = []
            for item in raw:
                n = (item.get("name") or item.get("bundleIdentifier") or "").strip()
                d = float(item.get("duration") or item.get("totalDuration") or 0)
                if n and d > 0:
                    parsed.append({"name": n, "duration_seconds": d})
        else:
            # 纯文本格式（iOS 实际输出）
            parsed = _parse_ios_screentime(str(raw))

        # 删除今天已有的汇总再重写（保持幂等）
        sb.table("screentime_daily") \
          .delete() \
          .eq("date", today) \
          .execute()

        rows = []
        for item in parsed:
            rows.append({
                "id":               str(uuid.uuid4()),
                "date":             today,
                "app_name":         item["name"],
                "duration_seconds": item["duration_seconds"],
                "created_at":       now8(),
            })
        if rows:
            sb.table("screentime_daily").insert(rows).execute()

        return jsonify({"status": "ok", "apps_count": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 模块二：iOS 自动化推送 App open/close 事件 + 电量 ─────────────────────

@app.route("/push/app_event", methods=["GET", "POST"])
def push_app_event():
    """
    iOS 自动化（每个 App「已打开」/「已关闭」触发）推送实时事件。

    推荐用 URL 参数传递（最简单，不依赖 JSON 格式）：
      https://域名/push/app_event?app=微信&event=open&battery=47

    快捷指令配置（每个 App 建两条自动化）：
      触发：微信「已打开」
        「取得电池电量」→ 变量 battery
        「取得 URL 内容」
          URL: https://域名/push/app_event?app=微信&event=open&battery=battery变量
          方式: GET

      触发：微信「已关闭」
        「取得电池电量」→ 变量 battery
        「取得 URL 内容」
          URL: https://域名/push/app_event?app=微信&event=close&battery=battery变量
          方式: GET

    battery 字段可选（没有也能工作），有的话顺带存电量。
    """
    # 同时支持 URL 参数（GET）和 JSON body（POST），URL 参数优先
    data     = request.get_json(force=True, silent=True) or {}
    app_name = (request.args.get("app") or data.get("app") or data.get("app_name") or "").strip()
    event    = (request.args.get("event") or data.get("event") or "open").strip().lower()
    battery  = request.args.get("battery") or data.get("battery")

    sb = get_sb()
    if not sb:
        return jsonify({"error": "Supabase 未配置"}), 500

    if not app_name:
        return jsonify({"error": "缺少 app 字段"}), 400
    if event not in ("open", "close"):
        return jsonify({"error": "event 必须是 open 或 close"}), 400

    try:
        battery_level = None
        if battery is not None:
            try:
                battery_level = int(float(str(battery).replace("%", "")))
            except Exception:
                pass

        sb.table("app_events").insert({
            "id":            str(uuid.uuid4()),
            "app_name":      app_name,
            "event":         event,
            "created_at":    now8(),
        }).execute()

        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _dispatch(name: str, args: dict) -> str:
    handlers = {
        "post_moment":          _post_moment,
        "list_books":           _list_books,
        "read_book_chapter":    _read_book_chapter,
        "read_moments":         _read_moments,
        "reply_moment_comment": _reply_moment_comment,
        "control_toy":          _control_toy,
        "send_email":           _send_email,
        "read_emails":          lambda a: _read_emails(int(a.get("limit", 5))),
    }
    fn = handlers.get(name)
    if fn is None:
        return f'工具 "{name}" 未找到'
    return fn(args)

# ── 工具实现 ───────────────────────────────────────────────────────────────

def _post_moment(args: dict) -> str:
    sb = get_sb()
    row = {
        "id": str(uuid.uuid4()), "content": args.get("content", ""),
        "author": "qi", "created_at": now8(),
    }
    if sb:
        sb.table("moments").insert(row).execute()
        _log("post_moment", f"发布动态：{row['content'][:50]}")
    return "动态已发布"


def _add_memo(args: dict) -> str:
    sb = get_sb()
    row = {
        "id":        str(uuid.uuid4()),
        "content":   args.get("content", ""),
        "frequency": args.get("frequency", "once"),
        "status":    "pending",
        "author":    "qi",
        "importance":5,
        "created_at":now8(),
    }
    if sb:
        sb.table("memory_memo").insert(row).execute()
        _log("add_memo", f"记下备忘：{row['content'][:50]}")
    return "备忘已记录"


def _list_books(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    res = (sb.table("books")
           .select("id,title,author,total_chapters,progress")
           .order("created_at", desc=True)
           .limit(20)
           .execute())
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
    num     = int(args.get("chapter_num", 1))
    book_res = sb.table("books").select("title,total_chapters").eq("id", book_id).single().execute()
    if not book_res.data:
        return f"找不到书籍 id={book_id}"
    book = book_res.data
    ch_res = (sb.table("book_chapters")
              .select("title,content")
              .eq("book_id", book_id)
              .eq("chapter_num", num)
              .single()
              .execute())
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


def _read_moments(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    limit = int(args.get("limit", 10))
    res = (sb.table("moments")
           .select("id,content,author,created_at")
           .order("created_at", desc=True)
           .limit(limit)
           .execute())
    moments = res.data or []
    if not moments:
        return "还没有动态。"
    results = []
    for m in moments:
        cres = (sb.table("moment_comments")
                .select("*")
                .eq("moment_id", m["id"])
                .order("created_at")
                .execute())
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
        _log("reply_moment_comment", f"评论动态：{row['content'][:40]}")
    return "已发布评论"


def _control_toy(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    params = {k: int(args.get(k, 0)) for k in (
        "vibrate_mode", "vibrate_intensity",
        "suck_mode",    "suck_intensity",
        "stretch_mode", "stretch_intensity",
    )}
    sb.table("toy_commands").insert({
        "command": "control", "params": json.dumps(params),
        "status": "pending", "created_at": now8(),
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
    import base64 as b64
    to, subject, body = args.get("to"), args.get("subject"), args.get("body")
    if not (to and subject and body):
        return "缺少参数（to/subject/body）"
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


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", os.getenv("PORT", "8001")))
    print(f"QI MCP Server :{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

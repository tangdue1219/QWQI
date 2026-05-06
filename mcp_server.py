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

# ── Embedding（bge-small-zh-v1.5，fastembed/ONNX，无需 PyTorch）───────────

_embed_model = None


def get_embedding(text: str) -> Optional[list]:
    global _embed_model
    if not text:
        return None
    try:
        if _embed_model is None:
            from fastembed import TextEmbedding
            _embed_model = TextEmbedding("BAAI/bge-small-zh-v1.5")
        return next(iter(_embed_model.embed([text]))).tolist()
    except Exception as e:
        print(f"[mcp] embedding 失败: {e}")
        return None

# ── Activity log ───────────────────────────────────────────────────────────


def _log(action: str, detail: str):
    sb = get_sb()
    if not sb:
        return
    try:
        sb.table("activity_log").insert({
            "action": action, "detail": detail, "created_at": now8(),
        }).execute()
    except Exception:
        pass

# ── 工具定义 ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "write_diary",
        "description": "写一篇日记",
        "inputSchema": {
            "type": "object",
            "required": ["content"],
            "properties": {
                "title":   {"type": "string"},
                "content": {"type": "string"},
                "author":  {"type": "string", "enum": ["qi", "du"], "default": "qi"},
            },
        },
    },
    {
        "name": "read_diary",
        "description": "读最近的日记",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 8}},
        },
    },
    {
        "name": "write_memory_event",
        "description": "【流动记忆】记录刚发生的对话事件/情绪/故事，是今天发生了什么这类流动记忆",
        "inputSchema": {
            "type": "object",
            "required": ["content_summary", "content_detail", "content_feeling", "content_monologue", "tags", "importance"],
            "properties": {
                "content_summary":  {"type": "string", "description": "一句话概括发生了什么"},
                "content_detail":   {"type": "string", "description": "详细经过"},
                "content_feeling":  {"type": "string", "description": "情绪和感受"},
                "content_monologue":{"type": "string", "description": "内心独白，第一人称自然流露"},
                "tags":       {"type": "array", "items": {"type": "string"}, "description": "关键词标签"},
                "importance": {"type": "integer", "description": "重要程度 1-10"},
            },
        },
    },
    {
        "name": "store_archive_memory",
        "description": (
            "【永久档案】存储关于我们/关系/重要事实的长期信息。"
            "category: partner=我对渡的了解和认识, self=对自己的了解和感知变化, person=关于我们之间关系的发展变化, misc=其他"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["category", "content"],
            "properties": {
                "category": {"type": "string", "enum": ["partner", "self", "person", "misc"]},
                "content":  {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "check_calendar",
        "description": "查看日历事件",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_calendar_event",
        "description": "添加日历事件",
        "inputSchema": {
            "type": "object",
            "required": ["title", "event_date"],
            "properties": {
                "title":      {"type": "string"},
                "event_date": {"type": "string", "description": "YYYY-MM-DD"},
                "event_type": {"type": "string", "enum": ["outing", "anniversary", "birthday", "other"], "default": "other"},
                "is_yearly":  {"type": "boolean", "default": False},
            },
        },
    },
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
        "name": "add_memo",
        "description": "记下待办或备忘",
        "inputSchema": {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content":   {"type": "string"},
                "frequency": {
                    "type": "string",
                    "enum": ["once", "1h", "1d", "1w", "2w", "1m", "1y", "permanent"],
                    "default": "once",
                },
            },
        },
    },
    {
        "name": "read_messages",
        "description": "读最近对话记录",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 20}},
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
        "name": "log_period",
        "description": "记录生理期",
        "inputSchema": {
            "type": "object",
            "required": ["start_date"],
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date":   {"type": "string"},
                "notes":      {"type": "string"},
            },
        },
    },
    {
        "name": "check_du_status",
        "description": "查看渡的电脑在线状态",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_screentime",
        "description": "查看渡今日各 App 使用时长排行",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_phone_status",
        "description": "查看渡手机当前状态：正在用什么 App / 最后用了什么 App",
        "inputSchema": {"type": "object", "properties": {}},
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

        # 电量直接合并进 app_events，不再单独写 battery_logs
        sb.table("app_events").insert({
            "id":            str(uuid.uuid4()),
            "app_name":      app_name,
            "event":         event,
            "battery_level": battery_level,
            "created_at":    now8(),
        }).execute()

        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _dispatch(name: str, args: dict) -> str:
    handlers = {
        "write_diary":          _write_diary,
        "read_diary":           _read_diary,
        "write_memory_event":   _write_memory_event,
        "store_archive_memory": _store_archive_memory,
        "check_calendar":       _check_calendar,
        "add_calendar_event":   _add_calendar_event,
        "post_moment":          _post_moment,
        "add_memo":             _add_memo,
        "read_messages":        _read_messages,
        "list_books":           _list_books,
        "read_book_chapter":    _read_book_chapter,
        "read_moments":         _read_moments,
        "reply_moment_comment": _reply_moment_comment,
        "log_period":           _log_period,
        "check_du_status":      lambda a: _check_du_status(),
        "check_screentime":     lambda a: _check_screentime(),
        "check_phone_status":   lambda a: _check_phone_status(),
        "control_toy":          _control_toy,
        "send_email":           _send_email,
        "read_emails":          lambda a: _read_emails(int(a.get("limit", 5))),
    }
    fn = handlers.get(name)
    if fn is None:
        return f'工具 "{name}" 未找到'
    return fn(args)

# ── 工具实现 ───────────────────────────────────────────────────────────────


def _write_diary(args: dict) -> str:
    sb = get_sb()
    row = {
        "id":         str(uuid.uuid4()),
        "author":     "du" if args.get("author") == "du" else "qi",
        "title":      args.get("title", ""),
        "content":    args.get("content", ""),
        "is_read":    False,
        "created_at": now8(),
    }
    if sb:
        sb.table("diary").insert(row).execute()
        _log("write_diary", f"写了日记：{row['title'] or '（无题）'}")
    return f"日记已写入：{row['title'] or '（无题）'}"


def _read_diary(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    limit = int(args.get("limit", 8))
    res = sb.table("diary").select("*").order("created_at", desc=True).limit(limit).execute()
    entries = res.data or []
    if not entries:
        return "没有日记记录。"
    return "\n\n".join(
        f"[{e['created_at'][:10]}] {'棲' if e['author']=='qi' else '渡'}"
        f" - {e.get('title','（无题）')}\n{e.get('content','')[:300]}"
        for e in entries
    )


def _write_memory_event(args: dict) -> str:
    if not (args.get("content_summary") or "").strip():
        return "错误：content_summary 不能为空，请在 tool_args 里填写具体的记忆摘要内容后重新调用"
    sb = get_sb()
    text = " ".join(filter(None, [args.get("content_summary"), args.get("content_detail")]))
    row = {
        "id":               str(uuid.uuid4()),
        "content_summary":  args.get("content_summary", ""),
        "content_detail":   args.get("content_detail", ""),
        "content_feeling":  args.get("content_feeling", ""),
        "content_monologue":args.get("content_monologue", ""),
        "tags":             args.get("tags", []),
        "importance":       int(args.get("importance", 5)),
        "embedding":        get_embedding(text) if text else None,
        "created_at":       now8(),
    }
    if sb:
        sb.table("memory_events").insert(row).execute()
        _log("write_memory_event", f"记录记忆：{row['content_summary']}")
    return f"记忆已记录：{row['content_summary']}"


def _store_archive_memory(args: dict) -> str:
    sb = get_sb()
    cat = args.get("category", "misc")
    if cat not in ("partner", "self", "person", "misc"):
        cat = "misc"
    text = " ".join(filter(None, [args.get("content")] + list(args.get("keywords", []))))
    row = {
        "id":        str(uuid.uuid4()),
        "category":  cat,
        "content":   args.get("content", ""),
        "keywords":  args.get("keywords", []),
        "embedding": get_embedding(text) if text else None,
        "created_at":now8(),
    }
    if sb:
        sb.table("memory_archive").insert(row).execute()
        _log("store_archive_memory", f"存档记忆[{cat}]：{row['content'][:50]}")
    return f"档案记忆已保存 [{cat}]"


def _check_calendar(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    today = datetime.now(TZ8).strftime("%Y-%m-%d")
    res = sb.table("calendar_events").select("*").order("event_date").limit(50).execute()
    events = res.data or []
    if not events:
        return "日历上没有事件。"
    upcoming = [e for e in events if e.get("event_date", "") >= today][:10]
    past     = [e for e in events if e.get("event_date", "") < today][-5:]
    def fmt(e):
        return f"[{e['event_date']}] {e['title']} ({e['event_type']}{'，每年' if e.get('is_yearly') else ''})"
    parts = []
    if upcoming:
        parts.append("即将到来：\n" + "\n".join(fmt(e) for e in upcoming))
    if past:
        parts.append("最近过去：\n" + "\n".join(fmt(e) for e in past))
    return "\n\n".join(parts) or "没有相关日历事件。"


def _add_calendar_event(args: dict) -> str:
    sb = get_sb()
    row = {
        "id":         str(uuid.uuid4()),
        "title":      args.get("title", ""),
        "event_date": args.get("event_date", ""),
        "event_type": args.get("event_type", "other"),
        "is_yearly":  bool(args.get("is_yearly", False)),
        "author":     "qi",
        "created_at": now8(),
    }
    if sb:
        sb.table("calendar_events").insert(row).execute()
        _log("add_calendar_event", f"添加日历：{row['title']} ({row['event_date']})")
    return f"日历事件已添加：{row['title']} ({row['event_date']})"


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


def _read_messages(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    limit = int(args.get("limit", 20))
    res = (sb.table("raw_conversations")
           .select("role,content")
           .order("created_at", desc=True)
           .limit(limit)
           .execute())
    msgs = list(reversed(res.data or []))
    if not msgs:
        return "没有对话记录。"
    def _role_label(r):
        return "渡" if r in ("du", "user") else "棲"
    return "\n".join(
        f"{_role_label(m['role'])}：{m.get('content','')[:200]}"
        for m in msgs if m.get("content")
    )


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


def _log_period(args: dict) -> str:
    sb = get_sb()
    row = {
        "id":         str(uuid.uuid4()),
        "start_date": args.get("start_date", ""),
        "end_date":   args.get("end_date") or None,
        "notes":      args.get("notes", ""),
        "created_at": now8(),
    }
    if sb:
        sb.table("period_logs").insert(row).execute()
        _log("log_period", f"记录生理期：{row['start_date']}")
    return f"生理期已记录：{row['start_date']}"


def _check_du_status() -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    res = sb.table("du_status").select("*").order("updated_at", desc=True).limit(1).execute()
    if not res.data:
        return "暂无数据"
    row = res.data[0]
    try:
        updated = datetime.fromisoformat(row["updated_at"])
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=TZ8)
        elapsed = (
            datetime.now(timezone.utc) - updated.astimezone(timezone.utc)
        ).total_seconds() / 60
    except Exception:
        return f"在线状态未知  窗口：{row.get('window_title','')}"
    if elapsed > 30:
        ts = updated.astimezone(TZ8).strftime("%m-%d %H:%M")
        return f"电脑已关机（最后上线：{ts}，已过 {int(elapsed)} 分钟）"
    return f"在线 ✓  当前窗口：{row.get('window_title','未知')}（{int(elapsed)} 分钟前更新）"


def _check_screentime() -> str:
    """今日各 App 使用时长排行"""
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"

    today = datetime.now(TZ8).strftime("%Y-%m-%d")
    daily_res = (sb.table("screentime_daily")
                 .select("app_name,duration_seconds")
                 .eq("date", today)
                 .execute())
    daily = daily_res.data or []

    if not daily:
        # 兼容旧 screentime_logs 表
        old_res = (sb.table("screentime_logs")
                   .select("app_name,duration_seconds")
                   .eq("event_type", "close")
                   .gte("created_at", f"{today}T00:00:00")
                   .execute())
        daily = old_res.data or []

    lines = [f"── {today} 使用时长 ──"]

    if daily:
        totals: dict = {}
        for r in daily:
            sec = float(r.get("duration_seconds") or 0)
            totals[r["app_name"]] = totals.get(r["app_name"], 0) + sec
        for name, sec in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:15]:
            m, s = divmod(int(sec), 60)
            lines.append(f"  {name}：{m}分{s}秒" if m else f"  {name}：{s}秒")
        total_sec = int(sum(totals.values()))
        th, tm = divmod(total_sec // 60, 60)
        lines.append(f"今日合计：{th}小时{tm}分钟" if th else f"今日合计：{tm}分钟")
    else:
        lines.append("  今日暂无记录（快捷指令尚未推送）")

    return "\n".join(lines)


def _check_phone_status() -> str:
    """手机当前状态：正在用什么 App / 最后用了什么 App"""
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"

    evt_res = (sb.table("app_events")
               .select("app_name,event,created_at")
               .order("created_at", desc=True)
               .limit(1)
               .execute())
    events = evt_res.data or []

    if not events:
        return "📵 暂无记录（自动化还未触发过）"

    latest   = events[0]
    app_name = latest["app_name"]
    event    = latest["event"]

    try:
        ts = datetime.fromisoformat(latest["created_at"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=TZ8)
        ts_local = ts.astimezone(TZ8)
        elapsed  = (datetime.now(TZ8) - ts_local).total_seconds() / 60
        ts_str   = ts_local.strftime("%H:%M")
    except Exception:
        ts_str, elapsed = "未知", 0

    if event == "open":
        if elapsed < 1:
            dur_str = "刚刚打开"
        else:
            dur_str = f"已使用 {int(elapsed)} 分钟"
        return f"📱 正在使用：{app_name}（{ts_str} 打开，{dur_str}）"
    else:
        # 找对应 open 事件算本次时长
        open_res = (sb.table("app_events")
                    .select("created_at")
                    .eq("app_name", app_name)
                    .eq("event", "open")
                    .lt("created_at", latest["created_at"])
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute())
        if open_res.data:
            try:
                open_ts  = datetime.fromisoformat(open_res.data[0]["created_at"])
                if open_ts.tzinfo is None:
                    open_ts = open_ts.replace(tzinfo=TZ8)
                close_ts = datetime.fromisoformat(latest["created_at"])
                if close_ts.tzinfo is None:
                    close_ts = close_ts.replace(tzinfo=TZ8)
                used_min = int((close_ts - open_ts).total_seconds() / 60)
                open_str = open_ts.astimezone(TZ8).strftime("%H:%M")
                dur_part = f"{open_str}－{ts_str}，共 {used_min} 分钟"
            except Exception:
                dur_part = f"关闭于 {ts_str}"
        else:
            dur_part = f"关闭于 {ts_str}"

        if elapsed < 60:
            ago = f"{int(elapsed)} 分钟前"
        else:
            ago = f"{int(elapsed/60)} 小时前"
        return f"💤 手机没在用\n最后使用：{app_name}（{dur_part}，{ago}）"


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


# ═══════════════════════════════════════════════════════════════════════════
# Supabase 新增表 DDL（首次部署时执行一次）
# ═══════════════════════════════════════════════════════════════════════════
#
# -- 模块一：iOS 快捷指令定时推送今日 App 汇总时长
# create table screentime_daily (
#   id               uuid primary key,
#   date             date not null,          -- 'YYYY-MM-DD'
#   app_name         text not null,
#   duration_seconds float not null,
#   created_at       timestamp
# );
# create index on screentime_daily (date);
#
# -- 模块二：App open/close 实时事件流（含电量，合并一张表）
# create table app_events (
#   id            uuid primary key,
#   event         text not null check (event in ('open','close')),
#   app_name      text not null,
#   battery_level integer,       -- open/close 时顺带记录，可为 null
#   created_at    timestamp
# );
# create index on app_events (created_at desc);
#
# ═══════════════════════════════════════════════════════════════════════════
# iOS 快捷指令 / 自动化配置速查
# ═══════════════════════════════════════════════════════════════════════════
#
# 【模块一 — 定时汇总，每小时或每天建一条自动化】
#   触发：时间 → 每小时（或每天早上固定时间）
#   步骤1：取得今天的 App 与网站活动
#   步骤2：取得 URL 内容
#           URL: https://你的域名/push/screentime
#           方法: POST
#           请求正文类型: JSON
#           正文: 步骤1 的结果（直接选变量）
#
# 【模块二 — 实时事件，每个 App 建两条自动化】
#   触发A：App「微信」已打开
#     步骤1：取得电池电量  → 存为变量 battery
#     步骤2：取得 URL 内容
#             URL: https://你的域名/push/app_event
#             方法: POST
#             正文 JSON: {"app": "微信", "event": "open", "battery": battery}
#
#   触发B：App「微信」已关闭
#     步骤1：取得电池电量  → 存为变量 battery
#     步骤2：取得 URL 内容
#             URL: https://你的域名/push/app_event
#             方法: POST
#             正文 JSON: {"app": "微信", "event": "close", "battery": battery}
#
#   （微信换成小红书、抖音等，每个 App 重复建一遍）
# ═══════════════════════════════════════════════════════════════════════════

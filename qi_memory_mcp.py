#!/usr/bin/env python3
"""
qi_memory_mcp.py — 棲的记忆 MCP 服务（给 Claude Code 用）

工具：
  get_memory_packet(user_message)   — 向量+关键词检索记忆，原文返回
  get_context_history(limit)        — 读 raw_conversations 最近 N 条
  store_memory_event(...)           — 存事件记忆
  store_archive_memory(...)         — 存档案记忆
  save_message(role, content)       — 存对话到 raw_conversations channel=cc

环境变量：
  SUPABASE_URL / SUPABASE_KEY
"""

import os, re, json
from datetime import datetime, timezone, timedelta
from math import exp
from dotenv import load_dotenv

load_dotenv()

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# ── 时间 ──────────────────────────────────────────────────────────────────────

TZ8 = timezone(timedelta(hours=8))

def now8() -> str:
    return datetime.now(TZ8).replace(tzinfo=None).isoformat(timespec="seconds")

# ── Supabase ──────────────────────────────────────────────────────────────────

_sb = None

def get_sb():
    global _sb
    if _sb is None:
        from supabase import create_client
        _sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _sb

# ── 向量嵌入 ──────────────────────────────────────────────────────────────────

_embed_model = None

def get_embedding(text: str) -> list[float]:
    global _embed_model
    if not text.strip():
        return []
    try:
        if _embed_model is None:
            from fastembed import TextEmbedding
            _embed_model = TextEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                cache_path=os.path.expanduser("~/.cache/fastembed"),
            )
        return list(_embed_model.embed([text]))[0].tolist()
    except Exception as e:
        print(f"[embedding] 失败: {e}", flush=True)
        return []

# ── 关键词提取 ────────────────────────────────────────────────────────────────

_STOPWORDS = set("的了吗呢啊哦哈嗯是在我你他她它们这那有没不也都还要会能可以就把被让给跟和与对从到说想看去做好吧呀嘛啦噢哇")

def _keywords(text: str, n: int = 5) -> list[str]:
    segs = re.findall(r'[一-鿿]{2,6}', text)
    seen, out = set(), []
    for s in segs:
        if all(c in _STOPWORDS for c in s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break
    return out

def _escape(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_")

# ── 时间衰减 ──────────────────────────────────────────────────────────────────

def _decay_score(e: dict) -> float:
    try:
        created = datetime.fromisoformat(e["created_at"])
        days = (datetime.now(TZ8).replace(tzinfo=None) - created).total_seconds() / 86400
        return float(e.get("importance", 5)) * exp(-0.693 * days / 30)
    except Exception:
        return float(e.get("importance", 5))

# ── 工具实现 ──────────────────────────────────────────────────────────────────

def _get_memory_packet(user_message: str) -> str:
    sb = get_sb()
    embedding = get_embedding(user_message)
    keywords = _keywords(user_message)

    # 事件记忆：向量 + 关键词 + tag
    event_seen: set = set()
    events: list[dict] = []

    if embedding:
        res = sb.rpc("match_memory_events", {
            "query_embedding": embedding,
            "match_threshold": 0.5,
            "match_count": 8,
        }).execute()
        for e in (res.data or []):
            if e.get("id") not in event_seen:
                events.append(e)
                event_seen.add(e["id"])

    for kw in keywords[:3]:
        safe = _escape(kw)
        res = sb.table("memory_events") \
            .select("id,content_summary,content_detail,content_feeling,content_monologue,tags,importance,created_at") \
            .or_(f"content_summary.ilike.%{safe}%,content_detail.ilike.%{safe}%") \
            .order("created_at", desc=True).limit(3).execute()
        for e in (res.data or []):
            if e.get("id") not in event_seen:
                events.append(e)
                event_seen.add(e["id"])

    if keywords:
        res = sb.table("memory_events") \
            .select("id,content_summary,content_detail,content_feeling,content_monologue,tags,importance,created_at") \
            .overlaps("tags", keywords) \
            .order("importance", desc=True).limit(5).execute()
        for e in (res.data or []):
            if e.get("id") not in event_seen:
                events.append(e)
                event_seen.add(e["id"])

    for e in events:
        e["_score"] = _decay_score(e)
    events.sort(key=lambda x: x["_score"], reverse=True)
    events = events[:6]

    # 档案记忆：向量 + 关键词
    arch_seen: set = set()
    archives: list[dict] = []

    if embedding:
        res = sb.rpc("match_memory_archive", {
            "query_embedding": embedding,
            "match_threshold": 0.5,
            "match_count": 3,
        }).execute()
        for a in (res.data or []):
            if a.get("id") not in arch_seen:
                archives.append(a)
                arch_seen.add(a["id"])

    for kw in keywords[:2]:
        safe = _escape(kw)
        res = sb.table("memory_archive") \
            .select("id,category,content,keywords,created_at") \
            .ilike("content", f"%{safe}%") \
            .order("created_at", desc=True).limit(2).execute()
        for a in (res.data or []):
            if a.get("id") not in arch_seen:
                archives.append(a)
                arch_seen.add(a["id"])

    archives = archives[:3]

    # 备忘录
    memos: list[dict] = []
    try:
        res = sb.table("memory_memo").select("content,importance") \
            .eq("status", "pending").order("importance", desc=True).limit(5).execute()
        memos = res.data or []
    except Exception:
        pass

    # 拼装输出
    parts: list[str] = []

    if events:
        lines = []
        for e in events:
            t = e.get("created_at", "")[:10]
            s = e.get("content_summary", "")
            d = e.get("content_detail", "")
            f = e.get("content_feeling", "")
            mono = e.get("content_monologue", "")
            lines.append(f"[{t}] {s}" + (f"\n细节：{d}" if d else "") + (f"\n感受：{f}" if f else "") + (f"\n独白：{mono}" if mono else ""))
        parts.append("## 事件记忆\n" + "\n\n".join(lines))

    if archives:
        lines = [f"[{a.get('category','')}] {a.get('content','')}" for a in archives]
        parts.append("## 档案记忆\n" + "\n".join(lines))

    if memos:
        lines = [m.get("content", "") for m in memos]
        parts.append("## 备忘\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "暂无相关记忆。"


def _get_context_history(limit: int = 20) -> str:
    sb = get_sb()
    res = sb.table("raw_conversations") \
        .select("role,content,created_at,channel") \
        .order("created_at", desc=True) \
        .limit(limit).execute()
    rows = list(reversed(res.data or []))
    if not rows:
        return "暂无对话记录。"
    lines = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if not content or content.upper() == "PASS":
            continue
        role = r.get("role", "")
        name = "渡" if role in ("user", "du") else "棲"
        ts = r.get("created_at", "")[:16].replace("T", " ")
        ch = r.get("channel", "")
        ch_tag = f"[{ch}]" if ch and ch != "web" else ""
        lines.append(f"{ts}{ch_tag} {name}: {content}")
    return "\n".join(lines) if lines else "暂无对话记录。"


def _store_memory_event(args: dict) -> str:
    sb = get_sb()
    content_summary   = args.get("content_summary", "")
    content_detail    = args.get("content_detail", "")
    content_feeling   = args.get("content_feeling", "")
    content_monologue = args.get("content_monologue", "")
    tags              = args.get("tags", [])
    importance        = float(args.get("importance", 5))

    text_for_embed = " ".join(filter(None, [content_summary, content_detail, content_feeling]))
    embedding = get_embedding(text_for_embed)
    if not embedding:
        return "TOOL_ERROR: embedding 失败"

    res = sb.table("memory_events").insert({
        "content_summary":   content_summary,
        "content_detail":    content_detail,
        "content_feeling":   content_feeling,
        "content_monologue": content_monologue,
        "tags":              tags,
        "embedding":         embedding,
        "importance":        importance,
        "decay_weight":      1.0,
        "created_at":        now8(),
    }).execute()
    new_id = res.data[0].get("id", "") if res.data else ""
    return f"TOOL_OK: 事件记忆已存入 id={new_id}"


def _store_archive_memory(args: dict) -> str:
    sb = get_sb()
    category = args.get("category", "misc")
    content  = args.get("content", "")
    keywords = args.get("keywords", [])

    if not content:
        return "TOOL_ERROR: content 不能为空"

    embedding = get_embedding(content)
    if not embedding:
        return "TOOL_ERROR: embedding 失败"

    sb.table("memory_archive").insert({
        "category":   category,
        "content":    content,
        "keywords":   keywords,
        "embedding":  embedding,
        "created_at": now8(),
    }).execute()
    return f"TOOL_OK: 档案记忆已存入 category={category}"


def _save_message(role: str, content: str) -> str:
    sb = get_sb()
    db_role = "user" if role in ("user", "du", "渡") else "assistant"
    sb.table("raw_conversations").insert({
        "role":       db_role,
        "content":    content,
        "channel":    "cc",
        "created_at": now8(),
    }).execute()
    return "TOOL_OK: 已存入对话"


# ── MCP Server ────────────────────────────────────────────────────────────────

server = Server("qi-memory")

TOOL_LIST = [
    types.Tool(
        name="get_memory_packet",
        description="根据当前话题，从棲的记忆库里检索相关的事件记忆、档案记忆和备忘，原文返回。",
        inputSchema={
            "type": "object",
            "properties": {
                "user_message": {"type": "string", "description": "当前话题或消息内容，用于检索相关记忆"},
            },
            "required": ["user_message"],
        },
    ),
    types.Tool(
        name="get_context_history",
        description="读取 raw_conversations 最近 N 条对话记录（含 cc/web/chat 渠道）。",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "description": "读取条数，默认20"},
            },
        },
    ),
    types.Tool(
        name="store_memory_event",
        description="把这段时间发生的事存入事件记忆库。",
        inputSchema={
            "type": "object",
            "required": ["content_summary"],
            "properties": {
                "content_summary":   {"type": "string", "description": "梗概"},
                "content_detail":    {"type": "string", "description": "细节"},
                "content_feeling":   {"type": "string", "description": "感受"},
                "content_monologue": {"type": "string", "description": "独白"},
                "tags":              {"type": "array", "items": {"type": "string"}},
                "importance":        {"type": "number", "default": 5, "description": "重要程度 1-10"},
            },
        },
    ),
    types.Tool(
        name="store_archive_memory",
        description="把一条长期事实存入档案记忆库。category 只能是 partner/self/person/misc。",
        inputSchema={
            "type": "object",
            "required": ["category", "content"],
            "properties": {
                "category": {"type": "string", "enum": ["partner", "self", "person", "misc"]},
                "content":  {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
        },
    ),
    types.Tool(
        name="save_message",
        description="把当前这轮对话存入 raw_conversations（channel=cc），让棲也能看到。",
        inputSchema={
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role":    {"type": "string", "enum": ["user", "assistant"], "description": "user=渡 assistant=棲"},
                "content": {"type": "string"},
            },
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOL_LIST


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "get_memory_packet":
            result = _get_memory_packet(arguments.get("user_message", ""))
        elif name == "get_context_history":
            result = _get_context_history(int(arguments.get("limit", 20)))
        elif name == "store_memory_event":
            result = _store_memory_event(arguments)
        elif name == "store_archive_memory":
            result = _store_archive_memory(arguments)
        elif name == "save_message":
            result = _save_message(arguments.get("role", "user"), arguments.get("content", ""))
        else:
            result = f'工具 "{name}" 不存在'
    except Exception as e:
        result = f"TOOL_ERROR: {e}"

    return [types.TextContent(type="text", text=result)]


async def main():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

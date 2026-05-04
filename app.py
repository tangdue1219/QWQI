"""
app.py — QI 云端服务

端点：
  POST /v1/chat/completions   Vertex AI Gemini 反代（OpenAI 兼容，支持流式）
  GET  /v1/models             模型列表
  GET  /tools                 MCP 工具列表
  POST /tools/call            MCP 工具执行
  POST /api/screentime        iOS 快捷指令上传手机使用记录
  GET  /                      健康检查

环境变量（Zeabur → Variables）：
  GATEWAY_API_KEY             接口鉴权 key（留空则不校验）
  GOOGLE_CREDENTIALS_JSON     Vertex 服务账号 JSON，支持原始 JSON 或 base64
  VERTEX_PROJECT_ID           GCP 项目 ID
  VERTEX_LOCATION             区域，默认 us-central1
  VERTEX_MODEL                默认 gemini-2.0-flash-001
  SUPABASE_URL                Supabase 项目地址
  SUPABASE_KEY                Supabase service_role key（MCP 工具读写用）
  EMAIL_USER                  Gmail 账号
  EMAIL_PASS                  Gmail App Password
  SMTP_HOST                   默认 smtp.gmail.com
  IMAP_HOST                   默认 imap.gmail.com
"""

import os, json, time, logging, base64, tempfile, smtplib
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from functools import wraps

from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── 配置 ──────────────────────────────────────────────────────────────

GATEWAY_KEY     = os.getenv("GATEWAY_API_KEY", "")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
VERTEX_PROJECT  = os.getenv("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL    = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")
EMAIL_USER      = os.getenv("EMAIL_USER", "")
EMAIL_PASS      = os.getenv("EMAIL_PASS", "")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
IMAP_HOST       = os.getenv("IMAP_HOST", "imap.gmail.com")

# ─── Vertex AI 凭据初始化 ──────────────────────────────────────────────

_creds_raw = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
if _creds_raw:
    try:
        try:
            decoded = base64.b64decode(_creds_raw.encode()).decode()
            json.loads(decoded)
            _creds_raw = decoded
        except Exception:
            pass
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tf.write(_creds_raw)
        tf.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
        log.info(f"Vertex 凭据已加载")
    except Exception as e:
        log.warning(f"Vertex 凭据加载失败: {e}")

# ─── Supabase 客户端（懒加载）─────────────────────────────────────────

_sb = None

def get_sb():
    global _sb
    if _sb is None and SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb

# ─── Auth ──────────────────────────────────────────────────────────────

def require_key(f):
    @wraps(f)
    def wrap(*a, **kw):
        if GATEWAY_KEY:
            tok = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
            if tok != GATEWAY_KEY:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*a, **kw)
    return wrap

# ─── 健康检查 ──────────────────────────────────────────────────────────

@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "qi-cloud"})

# ─── Vertex AI 反代 ────────────────────────────────────────────────────

@app.get("/v1/models")
@require_key
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": VERTEX_MODEL, "object": "model",
                  "created": int(time.time()), "owned_by": "vertex"}],
    })

@app.post("/v1/chat/completions")
@require_key
def chat_completions():
    import litellm
    litellm.vertex_project  = VERTEX_PROJECT
    litellm.vertex_location = VERTEX_LOCATION

    data      = request.get_json(force=True, silent=True) or {}
    messages  = data.get("messages", [])
    do_stream = data.get("stream", False)
    model_id  = f"vertex_ai/{VERTEX_MODEL}"

    kwargs = {"model": model_id, "messages": messages, "stream": do_stream}
    for k in ("temperature", "max_tokens", "max_completion_tokens",
              "top_p", "stop", "presence_penalty", "frequency_penalty"):
        if k in data:
            kwargs[k] = data[k]

    if do_stream:
        def _gen():
            try:
                for chunk in litellm.completion(**kwargs):
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                log.error(f"Vertex 流式错误: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(_gen()),
            mimetype="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache",
                     "Connection": "keep-alive"},
        )

    try:
        resp = litellm.completion(**kwargs)
        return jsonify(resp.model_dump())
    except Exception as e:
        log.error(f"Vertex 调用失败: {e}")
        return jsonify({"error": str(e)}), 500

# ─── MCP 工具定义 ──────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "check_du_status",
        "description": "查看渡目前的电脑状态（在线/离线，当前活动窗口标题）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_screentime",
        "description": "查看渡今日各 App 使用时长（手机屏幕时间）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "control_toy",
        "description": "控制连接的智能玩具（震动/吸吮/伸缩），数值 0 表示关闭该功能。",
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
        "description": "通过 Gmail 发送邮件。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "description": "收件人邮箱"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body":    {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "read_emails",
        "description": "读取 Gmail 最新邮件（默认最近 5 封）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 5},
            },
        },
    },
]

@app.get("/tools")
def list_tools():
    return jsonify({"tools": TOOL_DEFINITIONS})

# ─── MCP 工具执行路由 ──────────────────────────────────────────────────

@app.post("/tools/call")
def call_tool():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    args = data.get("arguments", data.get("args", {})) or {}

    if not name:
        return jsonify({"error": "name required"}), 400

    try:
        if name == "check_du_status":
            result = _check_du_status()
        elif name == "check_screentime":
            result = _check_screentime()
        elif name == "control_toy":
            result = _control_toy(args)
        elif name == "send_email":
            result = _send_email(args)
        elif name == "read_emails":
            result = _read_emails(int(args.get("limit", 5)))
        else:
            return jsonify({"error": f"未知工具：{name}"}), 404
        return jsonify({"result": result})
    except Exception as e:
        log.error(f"工具 {name} 执行失败: {e}")
        return jsonify({"error": str(e)}), 500

# ─── 工具实现 ──────────────────────────────────────────────────────────

CST = timezone(timedelta(hours=8))

def _check_du_status() -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    res = sb.table("du_status").select("*").order("updated_at", desc=True).limit(1).execute()
    if not res.data:
        return "暂无数据"
    row = res.data[0]
    updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
    elapsed_min = (datetime.now(timezone.utc) - updated).total_seconds() / 60
    if elapsed_min > 30:
        ts = updated.astimezone(CST).strftime("%m-%d %H:%M")
        return f"电脑已关机（最后上线：{ts}，已过 {int(elapsed_min)} 分钟）"
    return (f"在线 ✓  当前窗口：{row.get('window_title', '未知')}"
            f"（{int(elapsed_min)} 分钟前更新）")


def _check_screentime() -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    today = datetime.now(CST).strftime("%Y-%m-%d")
    res = (sb.table("screentime_logs")
             .select("app_name, duration_seconds")
             .eq("event_type", "close")
             .gte("created_at", f"{today}T00:00:00+08:00")
             .execute())
    if not res.data:
        return "今日暂无使用记录"
    totals: dict = {}
    for r in res.data:
        sec = float(r.get("duration_seconds") or 0)
        totals[r["app_name"]] = totals.get(r["app_name"], 0) + sec
    sorted_apps = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for app_name, sec in sorted_apps[:10]:
        m, s = divmod(int(sec), 60)
        lines.append(f"  {app_name}：{m}分{s}秒")
    total_min = int(sum(totals.values()) / 60)
    lines.append(f"\n今日合计：{total_min // 60}小时{total_min % 60}分钟")
    return "\n".join(lines)


def _control_toy(args: dict) -> str:
    sb = get_sb()
    if not sb:
        return "Supabase 未配置"
    params = {
        "vibrate_mode":      int(args.get("vibrate_mode", 0)),
        "vibrate_intensity": int(args.get("vibrate_intensity", 0)),
        "suck_mode":         int(args.get("suck_mode", 0)),
        "suck_intensity":    int(args.get("suck_intensity", 0)),
        "stretch_mode":      int(args.get("stretch_mode", 0)),
        "stretch_intensity": int(args.get("stretch_intensity", 0)),
    }
    sb.table("toy_commands").insert({
        "command": "control",
        "params": json.dumps(params),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }).execute()
    active = ", ".join(f"{k}={v}" for k, v in params.items() if v)
    return f"指令已发送：{active or '全部关闭'}"


def _send_email(args: dict) -> str:
    to, subject, body = args.get("to"), args.get("subject"), args.get("body")
    if not (to and subject and body):
        return "缺少参数（to / subject / body）"
    if not (EMAIL_USER and EMAIL_PASS):
        return "邮件配置未设置（EMAIL_USER / EMAIL_PASS）"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_USER
    msg["To"]      = to
    with smtplib.SMTP(SMTP_HOST, 587) as s:
        s.ehlo()
        s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_USER, [to], msg.as_string())
    return f"邮件已发送至 {to}"


def _read_emails(limit: int = 5) -> str:
    if not (EMAIL_USER and EMAIL_PASS):
        return "邮件配置未设置（EMAIL_USER / EMAIL_PASS）"
    import imaplib, email as _email
    from email.header import decode_header

    def _decode(s) -> str:
        if s is None:
            return ""
        parts = decode_header(str(s))
        out = []
        for b, enc in parts:
            if isinstance(b, bytes):
                out.append(b.decode(enc or "utf-8", errors="replace"))
            else:
                out.append(str(b))
        return "".join(out)

    with imaplib.IMAP4_SSL(IMAP_HOST) as M:
        M.login(EMAIL_USER, EMAIL_PASS)
        M.select("INBOX")
        _, ids = M.search(None, "ALL")
        uid_list = ids[0].split()
        recent = uid_list[-limit:] if uid_list else []
        results = []
        for uid in reversed(recent):
            _, msg_data = M.fetch(uid, "(RFC822)")
            msg = _email.message_from_bytes(msg_data[0][1])
            results.append(
                f"发件人：{_decode(msg['From'])}\n"
                f"主题：{_decode(msg['Subject'])}\n"
                f"时间：{msg['Date']}"
            )
    return ("\n---\n".join(results)) if results else "暂无邮件"

# ─── iOS 快捷指令：上传手机使用记录 ───────────────────────────────────
#
# POST /api/screentime
# 单条：{"app_name": "微信", "event_type": "open"}
#       {"app_name": "微信", "event_type": "close", "duration_seconds": 13}
# 批量：[{...}, {...}]

@app.post("/api/screentime")
def upload_screentime():
    sb = get_sb()
    if not sb:
        return jsonify({"error": "Supabase 未配置"}), 503
    raw = request.get_json(force=True, silent=True)
    if raw is None:
        return jsonify({"error": "无效数据"}), 400
    rows = raw if isinstance(raw, list) else [raw]
    insert = []
    for r in rows:
        if not r.get("app_name") or not r.get("event_type"):
            continue
        insert.append({
            "app_name":         r["app_name"],
            "event_type":       r["event_type"],
            "duration_seconds": r.get("duration_seconds"),
            "created_at":       r.get("created_at",
                                      datetime.utcnow().isoformat() + "Z"),
        })
    if not insert:
        return jsonify({"error": "无有效记录"}), 400
    sb.table("screentime_logs").insert(insert).execute()
    return jsonify({"ok": True, "inserted": len(insert)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info(f"QI Cloud :{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

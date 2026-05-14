"""Zeabur 入口文件"""
import os
import uvicorn
from mcp_server import asgi_app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)

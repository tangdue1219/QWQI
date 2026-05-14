"""Zeabur 入口文件，转发到 mcp_server"""
import os
import uvicorn
from mcp_server import starlette_app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

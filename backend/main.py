from __future__ import annotations

from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()


def main() -> None:
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
        log_level="info",
        timeout_keep_alive=600,  # 10分钟保持连接
    )


if __name__ == "__main__":
    main()

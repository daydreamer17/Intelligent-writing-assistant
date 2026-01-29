from __future__ import annotations

from dotenv import load_dotenv
import uvicorn

load_dotenv()


def main() -> None:
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

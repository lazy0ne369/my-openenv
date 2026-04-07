"""ASGI entrypoint expected by OpenEnv multi-mode deployment checks."""

import os

import uvicorn

from app import app as fastapi_app

app = fastapi_app


def main() -> None:
    """Run the FastAPI server from the required server.app entry point."""
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

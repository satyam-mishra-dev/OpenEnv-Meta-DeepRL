# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Shopops Environment.

This module creates an HTTP server that exposes the ShopopsEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi.responses import RedirectResponse

try:
    from ..models import ShopopsAction, ShopopsObservation
    from .shopOps_environment import ShopopsEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ShopopsAction, ShopopsObservation
    from server.shopOps_environment import ShopopsEnvironment


# Create the app with web interface and README integration
app = create_app(
    ShopopsEnvironment,
    ShopopsAction,
    ShopopsObservation,
    env_name="shopOps",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m shopOps.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn shopOps.server.app:app --workers 4
    """
    import os
    import uvicorn

    env_port = os.getenv("PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            pass

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

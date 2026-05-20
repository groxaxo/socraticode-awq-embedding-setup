#!/usr/bin/env python3
import asyncio
import os
import signal
import subprocess
import time
from contextlib import suppress

import httpx
from fastapi import FastAPI, Request, Response
import uvicorn


PUBLIC_HOST = os.getenv("PUBLIC_HOST", "0.0.0.0")
PUBLIC_PORT = int(os.getenv("PUBLIC_PORT", "8000"))
TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.getenv("TARGET_PORT", "18000"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "60"))
HEALTH_PATH = os.getenv("HEALTH_PATH", "/v1/models")
START_CMD = os.environ["START_CMD"]
START_TIMEOUT_SECONDS = int(os.getenv("START_TIMEOUT_SECONDS", "300"))

target_base = f"http://{TARGET_HOST}:{TARGET_PORT}"
app = FastAPI()
process: subprocess.Popen | None = None
last_activity = time.monotonic()
start_lock = asyncio.Lock()


async def target_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(target_base + HEALTH_PATH)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


async def ensure_started() -> None:
    global process
    async with start_lock:
        if process and process.poll() is None and await target_ready():
            return
        if not process or process.poll() is not None:
            process = subprocess.Popen(
                START_CMD,
                shell=True,
                preexec_fn=os.setsid,
            )

        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
            if await target_ready():
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"vLLM did not become ready within {START_TIMEOUT_SECONDS}s")


def stop_process() -> None:
    global process
    if not process or process.poll() is not None:
        process = None
        return

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    process = None


async def idle_reaper() -> None:
    while True:
        await asyncio.sleep(5)
        if process and process.poll() is not None:
            stop_process()
        if process and process.poll() is None:
            idle_for = time.monotonic() - last_activity
            if idle_for >= IDLE_SECONDS:
                stop_process()


@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(idle_reaper())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    stop_process()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request) -> Response:
    global last_activity
    last_activity = time.monotonic()
    await ensure_started()

    url = f"{target_base}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.request(
            request.method,
            url,
            content=body,
            headers=headers,
        )

    last_activity = time.monotonic()
    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=PUBLIC_HOST, port=PUBLIC_PORT, workers=1)

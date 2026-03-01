# dashboard_server.py
import asyncio
import json
import threading
import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pathlib import Path

from infer_realtime import run_realtime_inference

app = FastAPI()

latest = {
    "label": None,
    "valence": "—",
    "arousal": "—",
    "semantic": "Waiting for signal...",
    "confidence": 0.0,
    "timestamp": 0.0,
    "fs_eeg": None,
    "fs_ecg": None,
    "eeg_ch": None,
    "ecg_ch": None,
}

clients = set()
lock = threading.Lock()

def update_latest(payload: dict):
    with lock:
        latest.update(payload)

async def broadcaster():
    while True:
        await asyncio.sleep(0.2)  # 5 Hz 刷新（足够实时）
        with lock:
            msg = json.dumps(latest)
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

@app.on_event("startup")
async def startup_event():
    # 1) 推理线程
    def _infer_thread():
        run_realtime_inference(on_result=update_latest)

    t = threading.Thread(target=_infer_thread, daemon=True)
    t.start()

    # 2) 广播协程
    asyncio.create_task(broadcaster())

@app.get("/")
def index():
    html = Path("web/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            # 保持连接（前端不发也行）
            await ws.receive_text()
    except Exception:
        pass
    finally:
        clients.discard(ws)

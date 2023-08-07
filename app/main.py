import base64
import random
import cv2

from anyio import sleep
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from connection.connection import ConnectionManager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

manager = ConnectionManager()


def _encode_image(image):
    img_buffer = cv2.imencode(".jpg", image)[1]
    return base64.b64encode(img_buffer).decode("utf-8")


class FakePipeline:
    def __init__(self, user_id: str, conn_url: str):
        self._user_id = user_id
        self._conn_url = conn_url
        self._state = False

    def __call__(self):
        if random.random() > 0.9:
            self._state = True
        return self._state


@app.get("/")
async def get():
    with open("./static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conn_url: str):
    await manager.connect(websocket)
    cam = cv2.VideoCapture(0)
    try:
        while True:
            check, frame = cam.read()
            if not check:
                break
            await manager.send_message(user_id, "", frame, websocket)
            await sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        cam.release()
        cv2.destroyAllWindows()

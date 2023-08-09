import base64
import cv2

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_image(self, image, websocket: WebSocket):
        image_buffer = cv2.imencode(".jpg", image)[1]
        await websocket.send_bytes(base64.b64encode(image_buffer).decode("utf-8"))


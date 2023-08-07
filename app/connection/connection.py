import base64

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, state: bool, image, websocket: WebSocket):
        image_base64 = base64.b64encode(image).decode("utf-8")
        await websocket.send_json(
            data={
                "message": message,
                "state": state,
                "image": image_base64
            }
        )

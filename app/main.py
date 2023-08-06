import base64
import random
from anyio import sleep
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.pose import YoloPoseModel

object_model = YoloObjectDetector("../models/yolov8n.pt")

pose_model = YoloPoseModel("../models/yolov8n-pose.pt")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, state: bool, websocket: WebSocket):
        await websocket.send_json(data={"message": message, "state": state})

    async def send_image(self, image, websocket: WebSocket):
        await websocket.send_bytes(image)


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
            obj_results = object_model.predict(frame)
            pose_results = pose_model.predict(frame)
            if obj_results is not None:
                frame = object_model.draw_results(frame, obj_results)
            if pose_results is not None:
                frame = pose_model.draw_landmarks(frame, pose_results)
            encoded_image = _encode_image(frame)
            await manager.send_image(encoded_image, websocket)
            # await manager.send_message()
            await sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        cam.release()
        cv2.destroyAllWindows()


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket, user_id: str, conn_url: str):
#     await manager.connect(websocket)
#     pipe = FakePipeline(user_id, conn_url)
#     try:
#         while True:
#             if pipe():
#                 image = get_fake_image()
#                 # await manager.send_personal_message(f"Fall", False, websocket)
#                 await manager.send_image(image, websocket)
#             else:
#                 continue
#                 # await manager.send_personal_message(f"Not Fall", True, websocket)
#             await sleep(1)
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)


# @app.websocket("/ws/{user_id}")
# async def websocket_endpoint(websocket: WebSocket, user_id: str):
#     # check if connection exists/ connection posible/
#     await manager.connect(websocket)
#     if "video" in user_id:
#         pipeline = VideoPipeline(user_id, object_model, pose_model, pose_classifier)
#     try:
#         pipeline.run()
#         while True:
#             result = pipeline.next()
#             if result == "algo":
#                 await manager.send_personal_message(f"You wrote: {result}", websocket)
#             if "otra":
#                 manager.disconnect(websocket)
#                 return
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)

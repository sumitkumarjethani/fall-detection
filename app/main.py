from typing import Literal, Optional
import cv2
import pickle

from anyio import sleep
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from connection.connection import ConnectionManager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fall_detection.pose import YoloPoseModel
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.fall.pipeline import Pipeline

# Model Variables
yolo_pose_model = YoloPoseModel(model_path="../models/yolov8n-pose.pt")
yolo_object_model = YoloObjectDetector(model_path="../models/yolov8n.pt")
with open("../models/yolo-rf-pose-classifier.pkl", "rb") as f:
    pose_classifier = pickle.load(f)

# app variables
manager = ConnectionManager()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get():
    with open("./static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conn_url: str):
    await manager.connect(websocket)
    try:
        print(conn_url)
        cam = cv2.VideoCapture(int(conn_url) if conn_url == "0" else conn_url)

        pipeline = Pipeline(
            pose_model=yolo_pose_model,
            classification_model=pose_classifier,
            object_model=yolo_object_model,
        )

        c = 0
        while True:
            ok, input_frame = cam.read()
            if not ok:
                continue

            c = c + 1
            if c % 2 == 0:
                # Run pipeline
                output_frame, result_dict = pipeline._run(image=input_frame)
                print(result_dict)
                # Send output_frame via websocket
                await manager.send_image(output_frame, websocket)

            if c == 10:
                c = 0
            # await sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        raise e
    finally:
        cam.release()
        cv2.destroyAllWindows()
        return

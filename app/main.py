from typing import Literal, Optional
import cv2
import pickle

from anyio import sleep
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from connection.connection import ConnectionManager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fall_detection.pose import YoloPoseModel, MovenetModel
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.fall.pipeline import Pipeline

# "models/yolo_rf_pose_classifier_falldataset.pkl"

# model variables
# yolo_pose_model = YoloPoseModel(model_path="../models/yolov8n-pose.pt")
yolo_pose_model = MovenetModel("movenet_thunder")
yolo_object_model = YoloObjectDetector(model_path="../models/yolov8n.pt")
with open("../models/yolo_rf_pose_classifier_falldataset.pkl", "rb") as f:
    pose_classification_model = pickle.load(f)

# app variables
manager = ConnectionManager()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get():
    with open("./static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(html)


# class RTSPConnectionConfig(BaseModel):
#     ip: Optional[str] = "192.168.1.133"
#     port: int = 554
#     user: str = "falldetection"
#     password: str = "falldetection"

#     def get_url(self):
#         url = f"rtsp://{self.user}:{self.password}@{self.ip}:{str(self.port)}/stream1"
#         return url


# @app.websocket("/ws")
# async def websocket_endpoint(
#     websocket: WebSocket,
#     rtsp_conn: Optional[RTSPConnectionConfig] = Depends(),
# ):
#     await manager.connect(websocket)
#     try:
#         if rtsp_conn is not None:
#             cam = cv2.VideoCapture(rtsp_conn.get_url())
#         else:
#             cam = cv2.VideoCapture(0)

#         pipeline = Pipeline(
#             pose_model=yolo_pose_model,
#             classification_model=pose_classification_model,
#             object_model=yolo_object_model,
#         )

#         while True:
#             check, frame = cam.read()
#             if not check:
#                 break

#             output_frame = pipeline(frame)
#             await manager.send_image(output_frame, websocket)
#             await sleep(0.1)
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#     except Exception as e:
#         raise e
#     finally:
#         cam.release()
#         cv2.destroyAllWindows()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conn_url: str):
    await manager.connect(websocket)
    try:
        cam = cv2.VideoCapture(int(conn_url) if conn_url == "0" else conn_url)

        pipeline = Pipeline(
            pose_model=yolo_pose_model,
            classification_model=pose_classification_model,
            object_model=yolo_object_model,
        )

        while True:
            check, frame = cam.read()
            if not check:
                break

            output_frame = pipeline(frame)
            # output_frame
            await manager.send_image(output_frame, websocket)
            # await sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        raise e
    finally:
        cam.release()
        cv2.destroyAllWindows()

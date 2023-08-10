import cv2
import pickle

from anyio import sleep
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from connection.connection import ConnectionManager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fall_detection.pose import YoloPoseModel
from fall_detection.object_detection import YoloObjectDetector
from fall_detection.fall.pipeline import Pipeline


# model variables
yolo_pose_model = YoloPoseModel(model_path="C:/Users/sumit/OneDrive/Escritorio/models/yolov8n-pose.pt")
yolo_object_model = YoloObjectDetector(model_path="C:/Users/sumit/OneDrive/Escritorio/models/yolov8n.pt")

with open("C:/Users/sumit/OneDrive/Escritorio/models/yolo_rf_pose_clasifier_model.pkl", "rb") as f:
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conn_url: str):
    await manager.connect(websocket)
    try:
        cam = cv2.VideoCapture(int(conn_url) if len(conn_url)==1 else conn_url)
        
        pipeline = Pipeline(
            pose_model=yolo_pose_model,
            classification_model=pose_classification_model,
            object_model=yolo_object_model
        )

        while True:
            check, frame = cam.read()
            if not check:
                break

            output_frame = pipeline._run(frame)
            await manager.send_image(output_frame, websocket)
            await sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        raise e
    finally:
        cam.release()
        cv2.destroyAllWindows()

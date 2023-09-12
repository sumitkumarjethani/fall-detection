import cv2
import pickle

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from connection.connection import ConnectionManager
from notification.notification import NotificationManager
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
connection_manager = ConnectionManager()
notificaton_manager = NotificationManager()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get():
    with open("./static/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_email: str, conn_url: str):
    await connection_manager.connect(websocket)
    try:
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
                print("Not OK")
                break

            c = c + 1
            if c % 2 == 0:
                # Run pipeline
                output_frame, result_dict = pipeline._run(image=input_frame)
                print(result_dict)

                # Send output_frame via websocket
                await connection_manager.send_data(
                    output_frame, result_dict["detection"], websocket
                )

                # Send output fall frame via email noification
                if result_dict["detection"] == 1:
                    notificaton_manager.send_notification(
                        user_email, result=None, image=output_frame
                    )

            if c == 10:
                c = 0
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(e)
    finally:
        cam.release()
        cv2.destroyAllWindows()
        return

import os


def download_yolo_pose(output_path):
    try:
        os.system(
            "wget "
            + "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt"
            + " -O "
            + output_path
        )
    except Exception as e:
        raise Exception(f"could not download yolo pose: {e}")

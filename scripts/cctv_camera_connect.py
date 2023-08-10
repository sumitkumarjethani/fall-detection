import cv2

ipaddr = "192.168.1.133:554"
user = "falldetection"
password = "falldetection"

url = f"rtsp://{user}:{password}@{ipaddr}/stream1"

# "rtsp://root:pass@192.168.0.91:554/axis-media/media.amp"
if __name__ == "__main__":
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

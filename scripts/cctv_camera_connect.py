import cv2

# ipaddr = "192.168.1.133:554"
# user = "falldetection"
# password = "falldetection"
# url = f"rtsp://falldetection:falldetection@192.168.1.133:554/stream1"


ipaddr = "192.168.1.140:554"
user = "falldetection"
password = "falldetection"
url = f"rtsp://{user}:{password}@{ipaddr}/stream1"

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

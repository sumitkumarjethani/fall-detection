import cv2

# ipaddr = "192.168.1.133:554"
# user = "falldetection"
# password = "falldetection"
# url = f"rtsp://falldetection:falldetection@192.168.1.133:554/stream1"

# url = f"rtsp://falldetection:falldetection@172.20.10.7:554/stream1"

# ipaddr = "172.20.10.7:554"
# user = "falldetection"
# password = "falldetection"
# url = f"rtsp://{user}:{password}@{ipaddr}/stream1"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

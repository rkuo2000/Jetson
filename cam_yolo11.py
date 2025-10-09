#!/home/user/venv/bin/python

import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
#model = YOLO("yolo11n_fp32.engine")
#model = YOLO("yolo11n_int8.engine")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

prev_frame_time = 0

while (cap.isOpened()):
    ret, frame = cap.read() 
    new_frame_time = time.time()

    results = model(frame)
    annotated_image = results[0].plot() #

    if (new_frame_time - prev_frame_time)> 0:
        fps = 1/ (new_frame_time - prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    cv2.putText(annotated_image, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("YOLO11", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#!pip install opencv-python
#!pip install ultralytics

import cv2
import time
from ultralytics import YOLO

Name = "yolo26" # yolo11, yolo12, yolo13, yolo26
Size = "n" # n,s,m,l,x
Type = "" # "", -pose, -seg, -sem, -obb, -cls

model_name = Name + Size + Type
model = YOLO(model_name+".pt") 

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

prev_frame_time = 0

while (cap.isOpened()):
    ret, frame = cap.read() 
    new_frame_time = time.time()

    results = model(frame)

    if (new_frame_time - prev_frame_time)> 0:
        fps = 1/ (new_frame_time - prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    if Type=="-cls":
        top1 = results[0].probs.top1
        classified_name = results[0].names[top1]
        annotated_img = results[0].orig_img
        cv2.putText(annotated_img, classified_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        annotated_img = results[0].plot() 
        cv2.putText(annotated_img, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow(model_name, annotated_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

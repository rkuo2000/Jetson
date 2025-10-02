#!/home/user/venv/bin/python

import os
import cv2
import time
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('kavir-v2-efficentnet.h5')
model.summary()

cap = cv2.VideoCapture(0)

prev_frame_time = 0

while (cap.isOpened()):
    ret, frame = cap.read() 
    new_frame_time = time.time()

    img = cv2.resize(frame, (224, 224))

    results = model(img)
    print(results)

    if (new_frame_time - prev_frame_time)> 0:
        fps = 1/ (new_frame_time - prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    #cv2.putText(annotated_image, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("EfficentNet-v2s", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

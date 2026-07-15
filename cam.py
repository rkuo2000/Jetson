import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1980)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while(cap.isOpened()):
    ret, frame = cap.read() 
    print(frame.shape)

    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

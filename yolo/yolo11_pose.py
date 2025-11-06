from ultralytics import YOLO

model = YOLO("yolo11s-pose.pt")

results = model("images/baseball1.jpg")

results[0].show()

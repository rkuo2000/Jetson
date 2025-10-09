from ultralytics import YOLO

model = YOLO("yolo12n-face.pt")

results = model("images/face.jpg")

results[0].show()

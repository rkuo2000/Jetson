from ultralytics import YOLO

model = YOLO("yolo12s.pt")

#results = model("https://ultralytics.com/images/bus.jpg")
results = model("images/bus.jpg")

results[0].show()

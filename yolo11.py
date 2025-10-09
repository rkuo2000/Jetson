from ultralytics import YOLO

model = YOLO("yolo11n.pt")

#model.export(format="onnx")
#model = YOLO("yolo11n.onnx")

#results = model("https://ultralytics.com/images/bus.jpg")
results = model("bus.jpg")

results[0].show()

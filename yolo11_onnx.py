from ultralytics import YOLO

#model = YOLO("yolo11n.pt")
#model.export(format="engine")

onnx_model = YOLO("yolo11n.onnx")

#results = onnx_model("https://ultralytics.com/images/bus.jpg")
results = onnx_model("bus.jpg")

for result in results:
    result.show()

from ultralytics import YOLO

# Load a YOLO11n PyTorch model
#model = YOLO("yolo11n.pt")

# Export the model to TensorRT
#model.export(format="engine")

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

#results = onnx_model("https://ultralytics.com/images/bus.jpg")
results = trt_model("bus.jpg")

for result in results:
    result.show()

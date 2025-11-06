from ultralytics import YOLO

# Load a YOLO11s PyTorch model
model = YOLO("yolo11s.pt")

# Export the model to TensorRT
model.export(format="engine") # FP32
#model.export(format="engine",int8=True) # INT8

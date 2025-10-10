### [Utralytics guides](https://docs.ultralytics.com/guides/nvidia-jetson/)
`python3 -m venv .yolo`<br>

#### Install Ultralytics Package
```
sudo apt update
sudo apt install python3-pip -y
pip install -U pip

pip install ultralytics[export]

reboot
```

#### Install PyTorch and Torchvision
`pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl`<br>
`pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl`<br>

#### Install cuSPARSELt to fix a dependency issue with torch 2.5.0
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

#### Install onnxruntime-gpu
`wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/e1e/9e3dc2f4d5551/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl`<br>

#### 
`pip install numpy==1.23.5`

---
### Example
```
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")
```

---
### [YOLO-Face](https://github.com/YapaLab/yolo-face)

**model** : [yolov12n-face.pt](https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt)<br>

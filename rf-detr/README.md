# [RF-DETR](https://github.com/roboflow/rf-detr)
RF-DETR: SOTA Real-Time Detection and Segmentation Model<br>
**Paper**: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)<br>
**Paper**: [RF-DETR Object Detection vs YOLOv12](https://arxiv.org/html/2504.13099v1)<br>

`pip install rfdetr`<br>

## [Run a Pre-Trained Model](https://rfdetr.roboflow.com/learn/pretrained/)
```
import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = get_model("rfdetr-base")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```
![](https://media.roboflow.com/rfdetr-docs/annotated_image_base.jpg)

## [Train an RF-DETR Model](https://rfdetr.roboflow.com/learn/train/)
### Dataset structure: COCO format
```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

### Fine-Tuning
```
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>
)
```

#### ONNX export
`pip install rfdetr[onnxexport]`<br>

```
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export()
```

## [Deploy a Trained RF-DETR Model](https://rfdetr.roboflow.com/learn/deploy/)

## [Benchmarks](https://rfdetr.roboflow.com/learn/benchmarks/)

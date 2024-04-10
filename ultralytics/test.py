import sys
sys.path.append("..\\ultralytics\\ultralytics")
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load an official Detect model
results = model.export(
#source="E:\\项目开发\\项目代码\\ultralytics\\ultralytics\\assets",
format = 'onnx',
opset = 12,
imgsz = (480,640)
)

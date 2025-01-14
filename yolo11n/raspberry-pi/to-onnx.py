from ultralytics import YOLO

# Load your model
model = YOLO('best.pt')

# Export to ONNX
model.export(format='onnx')
# YOLOv11n Object Detection on Raspberry Pi

This repository contains C++ code for running YOLOv11n object detection on Raspberry Pi using OpenCV and ONNX runtime. The application captures video from the Raspberry Pi camera and performs real-time object detection.

## Prerequisites

- Raspberry Pi
- Raspberry Pi Camera Module
- Ubuntu or Raspberry Pi OS

## Installation

### 1. Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade
```

### 2. Install Dependencies

```bash
# Install build essentials and CMake
sudo apt-get install -y build-essential cmake

# Install OpenCV
sudo apt-get install -y libopencv-dev
```

### 3. Model Conversion

Before running the C++ application, you need to convert your PyTorch model (.pt) to ONNX format:

1. On your development machine (not Raspberry Pi), install the required Python packages:

```bash
pip install ultralytics opencv-python
```

2. Convert your model using Python:

```python
from ultralytics import YOLO

# Load your model
model = YOLO('best.pt')

# Export to ONNX
model.export(format='onnx')
```

3. Transfer the generated `best.onnx` file to your Raspberry Pi.

### 4. Create Classes File

Create a text file named `classes.txt` containing your class names, one per line. For example:

```
class1
class2
class3
```

### 5. Building the Application

1. Clone or copy the source code to your Raspberry Pi
2. Navigate to the project directory
3. Compile the code:

```bash
g++ -o yolo_detector yolo_detector.cpp `pkg-config --cflags --libs opencv4`
```

## Usage

1. Make sure your camera module is properly connected and enabled
2. Place your `best.onnx` and `classes.txt` files in the same directory as the executable
3. Run the application:

```bash
./yolo_detector
```

The application will:

- Open a window showing the camera feed
- Display bounding boxes around detected objects
- Show class labels and confidence scores
- Press 'q' to quit the application

## File Structure

```
├── yolo_detector.cpp
├── best.onnx
├── classes.txt
└── README.md
```

## Troubleshooting

1. If the camera doesn't open:

   - Check if the camera is enabled: `sudo raspi-config`
   - Verify camera connection
   - Try restarting the Raspberry Pi
2. If you get OpenCV errors:

   - Verify OpenCV installation: `pkg-config --modversion opencv4`
   - Reinstall OpenCV if necessary
3. If the model doesn't load:

   - Verify ONNX file path
   - Check if the ONNX file was converted correctly
   - Ensure classes.txt matches your model's classes

## Performance Notes

- The application runs on CPU using OpenCV's DNN module
- Performance will vary depending on your Raspberry Pi model
- For better performance:
  - Consider reducing input resolution
  - Adjust confidence threshold
  - Use a smaller model like YOLOv11n

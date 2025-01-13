import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import logging

# Disable unnecessary logging from YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load COCO class names
with open("class.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video capture (use webcam or video file, here using webcam)
cap = cv2.VideoCapture(0)

count = 0
prev_time = 0

while True:
    frame = picam2.capture_array()
    
    count += 1
    if count % 1 != 0:
        continue
    frame = cv2.flip(frame, -1)
    
    # Get the current time for FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, imgsz=240)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            if conf >= 0.4:  # Filter predictions with confidence below 40%
                c = class_names[class_id]
                x1, y1, x2, y2 = box
                # Print the result to the terminal
                print(f"Track ID: {track_id}, Class: {c}, Confidence: {conf:.2f}")
                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Display the FPS on the top-left corner
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the image in a window
    cv2.imshow("RGB", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

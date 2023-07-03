import os
import cv2
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

ROOT = Path(__file__).resolve().parent
sys.path.append(os.path.join(ROOT,"ByteTrack"))

from yolox.tracker.byte_tracker import BYTETracker, STrack


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes

MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection2.mkv')

f , s = 0, 0
colors = [ (0, 255, 255), (255,0,0), (0, 0, 255), (255,140,100), (20, 55, 95), (85,160,90), ]

cap = cv2.VideoCapture(SOURCE_PATH)
model = YOLO(MODEL_PATH)
model.fuse()

ID2CLASSES = model.names
classes = ID2CLASSES.values()

text_scale = 1.5
text_thickness = 2
line_thickness = 2

class ByteTrackArgument:
    track_thresh = 0.5
    track_buffer = 50
    match_thresh = 0.8
    aspect_ratio_thresh = 10.0
    min_box_area = 1.0
    mot20 = False

MIN_THRESHOLD = 0.001
INPUT_VIDEO_PATH = "samples/detection.mp4"


frame_id = 0

history = deque()

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLO
    results = model.predict(frame_rgb, iou = 0.4)
    frame = results[0].plot()
    # Display the frame
    cv2.imshow('Tracker', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

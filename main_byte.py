import os
import cv2
import random
import numpy as np
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import time



IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes


ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection2.mkv')

f , s = 0, 0.1
random.seed(41)
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
save_result = True
plot_basketball = False
trackers = [BYTETracker(ByteTrackArgument), BYTETracker(ByteTrackArgument), BYTETracker(ByteTrackArgument)]
frame_id = 0
results = []
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

    # Extract bounding box coordinates and class labels
    bboxes = results.xyxy[0].tolist()
    class_ids = results.pred[0].tolist()

    # Update the tracker with the bounding boxes
    
    # Visualize the tracked objects
    for bbox, id in zip():
        x1, y1, x2, y2 = bbox
        color = colors[id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{ID2CLASSES[class_ids[id]]} {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Tracker', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# if SOURCE_PATH.endswith(IMG_FORMATS):
#     try:
#         image = cv2.imread(SOURCE_PATH)
#     except Exception as e:
#         print(e, "\nExitting!!")

#     cv2.imshow("Ouput", image)
#     print("Press Q to quit")
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()


# else:
#     while True:
#         s = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model.predict(frame, iou= 0.2, verbose =False)
#         results = results[0]
#         detections = []
#         classes = []
        
#         # frame = results.plot(line_width = 1, font_size = 0.1)
#         fps = 1/(s-f)
#         f = s
#         cv2.putText(frame, str(fps), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
        
        
#         cv2.imshow("Output", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     cap.release()        
#     cv2.destroyAllWindows()



import os
import cv2
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from tracker.bytetrack import *

ROOT = Path(__file__).resolve().parent
sys.path.append(os.path.join(ROOT,"ByteTrack"))

from yolox.tracker.byte_tracker import BYTETracker, STrack

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes

MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection.mp4')

f , s = 0, 0
colors = [ (0, 255, 255), (255,0,0), (0, 0, 255), (255,140,100), (20, 55, 95), (85,160,90), ]

cap = cv2.VideoCapture(SOURCE_PATH)
model = YOLO(MODEL_PATH)
model.fuse()

ID2CLASSES = model.names
classes = ID2CLASSES.values() #        ['Boots', 'Gloves', 'Helmet', 'Hi-Vis Jacket', 'Person', 'Protective Glasses']

print(ID2CLASSES)
text_scale = 1.5
text_thickness = 2
line_thickness = 2

MIN_THRESHOLD = 0.001
INPUT_VIDEO_PATH = "samples/detection.mp4"

"""
Some shit
"""
# red
BOOT_COLOR_HEX = "#850101"
BOOT_COLOR = Color.from_hex_string(BOOT_COLOR_HEX)

# green
GLOVE_COLOR_HEX = "#00D4BB"
GLOVE_COLOR = Color.from_hex_string(GLOVE_COLOR_HEX)

#red
HELMET_COLOR_HEX = "#850101"
HELMET_COLOR = Color.from_hex_string(HELMET_COLOR_HEX)

# green
VEST_COLOR_HEX = "#00D4BB"
VEST_COLOR = Color.from_hex_string(VEST_COLOR_HEX)

# yellow
PERSON_COLOR_HEX = "#FFFF00"
PERSON_COLOR = Color.from_hex_string(PERSON_COLOR_HEX)

# yellow
GOGGLES_COLOR_HEX = "#FFFF00"
GOGGLES_COLOR = Color.from_hex_string(GOGGLES_COLOR_HEX)


COLORS = [
    BOOT_COLOR,
    GLOVE_COLOR,
    HELMET_COLOR,
    VEST_COLOR,
    PERSON_COLOR,
    GOGGLES_COLOR
]
THICKNESS = 4

tracker = BYTETracker(BYTETrackerArgs())

base_annotator = BaseAnnotator(
    colors=[
        BOOT_COLOR,
        GLOVE_COLOR,
        HELMET_COLOR,
        VEST_COLOR,
        PERSON_COLOR,
        GOGGLES_COLOR
    ], 
    thickness=THICKNESS)


height = 640
width = 640
frame_id = 0
results = []
history = deque()

# Process each frame in the video

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, iou = 0.4)
        detections = Detection.from_results(pred=results[0].boxes.data.detach().cpu().numpy(), names= ID2CLASSES)
        ## Filter detections by class

        helmet_detections = filter_detections_by_class(detections, class_name="Helmet")
        vest_detections = filter_detections_by_class(detections, class_name="Hi-Vis Jacket")
        person_detections = filter_detections_by_class(detections, class_name="Person")
        
        tracked_detections = helmet_detections + vest_detections + person_detections

        tracks = tracker.update(
            output_results= detections2boxes(detections= tracked_detections),
            img_info=frame.shape,
            img_size= frame.shape
        )
        
        tracked_detections = match_detections_with_tracks(detections= tracked_detections, tracks= tracks)
        helmet_detections = filter_detections_by_class(tracked_detections, class_name="Helmet")
        vest_detections = filter_detections_by_class(tracked_detections, class_name="Hi-Vis Jacket")
        person_detections = filter_detections_by_class(tracked_detections, class_name="Person")

        annotated_frame = frame.copy()
        annotated_frame = base_annotator.annotate(
            image=annotated_frame, 
            detections=tracked_detections
        )

        #frame = results[0].plot()
        cv2.imshow("Image", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


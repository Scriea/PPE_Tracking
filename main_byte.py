import os
import cv2
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from utils.helper import COLORS, BACKGROUND_COLORS, TEXT_COLOR
from tracker.bytetrack import *

ROOT = Path(__file__).resolve().parent
sys.path.append(os.path.join(ROOT,"ByteTrack"))

from yolox.tracker.byte_tracker import BYTETracker, STrack

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes

MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection2.mkv')

f , s = 0, 0
cap = cv2.VideoCapture(SOURCE_PATH)
model = YOLO(MODEL_PATH)
model.fuse()

ID2CLASSES = model.names
classes = ID2CLASSES.values() #        ['Boots', 'Gloves', 'Helmet', 'Hi-Vis Jacket', 'Person', 'Protective Glasses']

print(ID2CLASSES)

text_scale = 1.5
text_thickness = 1
line_thickness = 2

MIN_THRESHOLD = 0.001


person_tracker = BYTETracker(BYTETrackerArgs())
tracker = BYTETracker(BYTETrackerArgs())

base_annotator = BaseAnnotator(
    colors= COLORS,
    thickness=line_thickness
)

text_annotator = TextAnnotator(
    text_colors= TEXT_COLOR, 
    background_color= BACKGROUND_COLORS, 
    text_thickness= text_thickness
)

height = 640
width = 640
frame_id = 0
results = []
history = deque()

# Process each frame in the video

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, iou = 0.3, verbose = False)
        detections = Detection.from_results(pred=results[0].boxes.data.detach().cpu().numpy(), names= ID2CLASSES)
        ## Filter detections by class

        # helmet_detections = filter_detections_by_class(detections, class_name="Helmet")
        # vest_detections = filter_detections_by_class(detections, class_name="Hi-Vis Jacket")
        # person_detections = filter_detections_by_class(detections, class_name="Person")
        # glove_detections = filter_detections_by_class(detections, class_name="Gloves")
        # goggles_detections = filter_detections_by_class(detections, class_name="Protective Glasses")

        # tracked_detections = helmet_detections + vest_detections + glove_detections + goggles_detections

        tracks = tracker.update(
            output_results= detections2boxes(detections= detections),
            img_info=frame.shape,
            img_size= frame.shape
        )
        
        tracked_detections = match_detections_with_tracks(detections= detections, tracks= tracks)
        # person_detections = match_detections_with_tracks(detections= person_detections, tracks=person_tracks)

        annotated_frame = frame.copy()
        annotated_frame = base_annotator.annotate(
            image=annotated_frame, 
            detections=tracked_detections
        )
        # annotated_frame = base_annotator.annotate(
        #     image=annotated_frame, 
        #     detections=person_detections,
        # )
        annotated_frame = text_annotator.annotate(
            image=annotated_frame, 
            detections= tracked_detections,
        )

        frame = results[0].plot()
        r = 800 / frame.shape[1]
        dim = (800, int(frame.shape[0] * r))
        print(tracked_detections)
        cv2.imshow("Image", cv2.resize(annotated_frame, dim, cv2.INTER_AREA))

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


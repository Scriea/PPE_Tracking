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

MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','yolov8n_e100_newdataset.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection.mp4')
SOURCE_URL = "rtsp://rtsp:Rtsp1234@158.0.17.112:554/streaming/channels/1"

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
ret = True
# Process each frame in the video

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        r = 800 / frame.shape[1]
        dim = (800, int(frame.shape[0] * r))
        results = model.predict(frame, conf =0.1, iou = 0.3)               # verbose = False, stops printing logs
        detections = Detection.from_results(pred=results[0].boxes.data.detach().cpu().numpy(), names= ID2CLASSES)
        

        # helmet_detections = filter_detections_by_class(detections, class_name="Helmet")
        # vest_detections = filter_detections_by_class(detections, class_name="Hi-Vis Jacket")
        # person_detections = filter_detections_by_class(detections, class_name="Person")
        # glove_detections = filter_detections_by_class(detections, class_name="Gloves")
        # goggles_detections = filter_detections_by_class(detections, class_name="Protective Glasses")

        # tracked_detections = helmet_detections + vest_detections + glove_detections + goggles_detections

        output_results= detections2boxes(detections= detections)

        # print("----------")
        # print(type(output_results), output_results)
        # print("----------")
        
        if len(output_results) > 0:
            tracks = tracker.update(
                output_results= output_results,
                img_info=frame.shape,
                img_size= frame.shape
            )
            
            tracked_detections = match_detections_with_tracks(detections= detections, tracks= tracks)
            annotated_frame = frame.copy()
            annotated_frame = base_annotator.annotate(
                image=annotated_frame, 
                detections=tracked_detections
            )
            annotated_frame = text_annotator.annotate(
                image=annotated_frame, 
                detections= tracked_detections,
            )
            frame = results[0].plot()
            print([detection.get_ids() for detection in tracked_detections])
            cv2.imshow("Image", cv2.resize(annotated_frame, dim, cv2.INTER_AREA))
        else:
            cv2.imshow("Image", cv2.resize(frame, dim, cv2.INTER_AREA))
        #cv.imshow("Image", results[0].plot()
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


import sys
import os
import PIL
import cv2
from ultralytics import YOLO
from pathlib import Path

ROOT = os.path.abspath(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from utils.helper import COLORS, BACKGROUND_COLORS, TEXT_COLOR
from tracker.bytetrack import *
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack


text_scale = 1.5
text_thickness = 1
line_thickness = 2
MIN_THRESHOLD = 0.001



base_annotator = BaseAnnotator(
    colors= COLORS,
    thickness=line_thickness
)

text_annotator = TextAnnotator(
    text_colors= TEXT_COLOR, 
    background_color= BACKGROUND_COLORS, 
    text_thickness= text_thickness
)


class Detector:
    def __init__(self,MODEL_PATH, tracker, width = 1024, height = 720, SOURCE_PATH = None) -> None:
        self.SOURCE_PATH = SOURCE_PATH
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()
        self.tracker = tracker
        self.ID2CLASSES = self.model.names
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.SOURCE_PATH) if self.SOURCE_PATH else None
        # if self.cap:
        #     self.cap.set(3, width)
        #     self.cap.set(4, height)
        
    def __del__(self):
        if self.cap:
            if self.cap.isOpened():
                self.cap.release()

    def change_stream(self, SOURCE_PATH):
        self.SOURCE_PATH = SOURCE_PATH
        try:
            self.cap.release()
        except Exception as e:
            print(e)
            pass
        self.cap = cv2.VideoCapture(self.SOURCE_PATH)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        
    def get_frame_size(self):
        return self.width, self.height

    def get_frame(self):
        if not self.cap:
            return False, None
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height), interpolation= cv2.INTER_LINEAR)
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, None
    
    def predict(self):
        ret = False
        frame = None
        if not self.cap:
            return False, None
        if self.cap.isOpened():
            ret, frame = self.cap.read()         
            if not ret:
                return ret, None
            results = self.model.predict(frame, conf =0.3, iou = 0.15)
            detections = detections = Detection.from_results(pred=results[0].boxes.data.detach().cpu().numpy(), names= self.ID2CLASSES)
            output_results= detections2boxes(detections= detections)
            if len(output_results) > 0:
                tracks = self.tracker.update(
                    output_results= output_results,
                    img_info=frame.shape,
                    img_size= frame.shape
                )
                
                tracked_detections = match_detections_with_tracks(detections= detections, tracks= tracks)
                #annotated_frame = frame.copy()
                frame = base_annotator.annotate(
                    image=frame, 
                    detections=tracked_detections
                )
                # frame = text_annotator.annotate(
                #     image=frame, 
                #     detections= tracked_detections,
                # )
                #frame = results[0].plot(line_width = 1, font_size = 0.1)
                # frame = cv2.resize(frame, (self.width, self.height), interpolation= cv2.INTER_LINEAR)

            frame = cv2.cvtColor(cv2.resize(frame, (self.width, self.height), interpolation= cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)

        return ret, frame 
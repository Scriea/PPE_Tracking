import os
import cv2
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(os.path.abspath(ROOT))

import time
import numpy as np
from collections import deque
from ultralytics import YOLO
from tracker.bytetrack import *
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any


# draw utilities
@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
        
    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


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
PERSON_COLOR_HEX = "#FFBF00"
PERSON_COLOR = Color.from_hex_string(PERSON_COLOR_HEX)

# yellow
GOGGLES_COLOR_HEX = "#FFFF00"
GOGGLES_COLOR = Color.from_hex_string(GOGGLES_COLOR_HEX)

TEXT_COLOR_HEX = "#FFFFFF"
TEXT_COLOR = Color.from_hex_string(TEXT_COLOR_HEX)

COLORS = [
    BOOT_COLOR,
    GLOVE_COLOR,
    HELMET_COLOR,
    VEST_COLOR,
    PERSON_COLOR,
    GOGGLES_COLOR
]

BACKGROUND_COLORS = [
    BOOT_COLOR,
    GLOVE_COLOR,
    HELMET_COLOR,
    VEST_COLOR,
    PERSON_COLOR,
    GOGGLES_COLOR
]


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

# geometry utilities
@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y
    
    @property
    def max_x(self) -> float:
        return self.x + self.width
    
    @property
    def max_y(self) -> float:
        return self.y + self.height
        
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)
    
    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding, 
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )
    
    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y




@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_rect(
                image= image, 
                rect= detection.rect,
                color=self.colors[detection.class_id],
                thickness= self.thickness
            )
        return annotated_image
    

@dataclass
class TextAnnotator:
    background_color: List[Color] 
    colors: List[Color]
    text_thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            # if tracker_id is not assigned skip annotation
            if detection.tracker_id is None:
                continue
            
            # calculate text dimensions
            s = str(detection.class_name) +str(detection.tracker_id) + ": "+  str(detection.confidence)
            size, _ = cv2.getTextSize(
                str(s), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                thickness=self.text_thickness)
            width, height = size
            # calculate text background position
            center_x, center_y = detection.rect.top_center.int_xy_tuple
            x = center_x + width // 2
            y = center_y + height // 2 + 10
            
            # draw background
            annotated_image = draw_filled_rect(
                image=annotated_image, 
                rect=Rect(x=x, y=y, width=width, height=height).pad(padding=5), 
                color=self.background_color)
            
            # draw text
            annotated_image = draw_text(
                image=annotated_image, 
                anchor=Point(x=x, y=y + height), 
                text= s, 
                color=self.colors[detection.class_id], 
                thickness=self.text_thickness)
        return annotated_image

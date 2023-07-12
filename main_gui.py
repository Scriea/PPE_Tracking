import os
from tkinter import *
import customtkinter
from ultralytics import YOLO
from utils.app import App
from utils.detector import Detector
from utils.general import get_stream_list

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('blue')

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from tracker.bytetrack import *
from yolox.tracker.byte_tracker import BYTETracker

"""
Paths

"""
ROOT = os.getcwd()
STREAM_SOURCE = os.path.join(ROOT,'stream.txt')
ROOT = os.getcwd();
MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','yolov8n_e100_newdataset.pt')


try:
    STREAM_LIST = get_stream_list(source= STREAM_SOURCE)
    SOURCE_PATH = STREAM_LIST[0]
except Exception as e:
    print(e)
    print("Please enter a video path/url in stream.txt")
    exit()

frame_width = 1200
frame_height = 800
tracker = BYTETracker(BYTETrackerArgs())
detector = Detector(MODEL_PATH, tracker=tracker, width= frame_width, height= frame_height)

"""
GUI

"""
app = App(
    detector=detector, 
    tracker= tracker,
    window_title= "PPE Detector",
    source= STREAM_SOURCE, 
    delay= 1
)
app.update()
app.mainloop()
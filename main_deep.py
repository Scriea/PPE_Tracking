import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tracker.deepsort import Tracker
import time



IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes


ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(ROOT, 'models', 'yolo','best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection2.mkv')

f , s = 0, 0.1
random.seed(41)
colors = [ (0, 255, 255), (255,0,0), (0, 0, 255), (255,140,100), (20, 55, 95), (85,160,90), ]
tracker = Tracker()
cap = cv2.VideoCapture(SOURCE_PATH)
model = YOLO(MODEL_PATH)
model.fuse()


if SOURCE_PATH.endswith(IMG_FORMATS):
    try:
        image = cv2.imread(SOURCE_PATH)
    except Exception as e:
        print(e, "\nExitting!!")

    cv2.imshow("Ouput", image)
    print("Press Q to quit")
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


else:
    while True:
        s = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, iou= 0.2, verbose =False)
        results = results[0]
        detections = []
        classes = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2, class_id = int(x1),int(y1),int(x2),int(y2), int(class_id)
            detections.append([x1, y1, x2, y2, score])
            classes.append(class_id)

        tracker.update(frame, detections)

        for i, track in enumerate(tracker.tracks):
            bbox, track_id = track.get_data()
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1,y1),(x2,y2),(colors[classes[i % len(classes)]]), 1)
        
        #bframe = results.plot(line_width = 1, font_size = 0.1)
        fps = 1/(s-f)
        f = s
        cv2.putText(frame, str(fps), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
        
        
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()        
    cv2.destroyAllWindows()



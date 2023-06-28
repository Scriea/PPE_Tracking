import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'mp4'  # include video suffixes


ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(ROOT, 'models', 'best.pt')
SOURCE_PATH = os.path.join(ROOT, 'samples', 'detection.mp4')

cap = cv2.VideoCapture()
model = YOLO(MODEL_PATH)


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
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()        
    cv2.destroyAllWindows()



# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
from ultralytics import YOLO
import cv2
import math
import time
from threading import Thread
import importlib.util

#add custom command
from gpiozero import LED
from time import sleep
#red = LED(12)
gpio = LED(17)

cap = cv2.VideoCapture(0)

#add custom command
model = YOLO("yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    


while True:
    success, img = cap.read()
    results = model.predict(img, conf = 0.20 )
    
    red.on()
    gpio.on()
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            org = [x1, y1]
            org1 = [x1-40, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, str(confidence), org1, font, fontScale, color, thickness)
            
            #add custom command
            if classNames[cls] == "mouse":
                gpio.on()
                sleep(1)
                gpio.off()
                sleep(1)


                
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

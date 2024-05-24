from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('./video/people.mp4') # for video

model = YOLO('./yolo_weights/yolov8n.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

mask = cv2.imread('./image/elev_mask.png')

# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103,161,296,160]
limitsDown = [527,489,735,489]
up_count = 0
down_count = 0
countUp_dict = {}
countDown_dict = {}

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    imgGraphic = cv2.imread('./image/graphics-1.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphic, (730,260))
    
    results = model(imgRegion, stream=True)
    
    detactions = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil(box.conf[0]*100) / 100
            cls = int(box.cls[0])
            
            currentClass = classNames[cls]
            
            if currentClass in ['person'] and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detactions = np.vstack((detactions, currentArray))
                
    resultTracker = tracker.update(detactions)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,0,255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,0,255), 5)
    
    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        # print(result)
        w, h = x2-x1, y2-y1
        id = int(id)
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{id}', (x1, y1-20), 
                               scale=2, thickness=3, offset=10)
        
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx, cy), 5, (0,0,255), cv2.FILLED)
        
        id_index = f'{id}'
        # Count up
        if (limitsUp[0]<cx<limitsUp[2]) and (limitsUp[1]<cy<limitsUp[1]+100):
            countUp_dict[id_index] = 1
        elif limitsUp[0]<cx<limitsUp[2] and cy<limitsUp[1] and id_index in countUp_dict:
            up_count += 1
            countUp_dict.pop(id_index)
            cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,255,0), 5)

        # Count down
        if (limitsDown[0]<cx<limitsDown[2]) and (limitsDown[1]-100<cy<limitsDown[1]):
            countDown_dict[id_index] = 1
        elif limitsDown[0]<cx<limitsDown[2] and cy>limitsDown[1] and id_index in countDown_dict:
            down_count += 1
            countDown_dict.pop(id_index)
            cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,255,0), 5)
    
    # cvzone.putTextRect(img, f'Count: {total_count}', (50, 50))
    cv2.putText(img, f"{up_count}", (935,345), cv2.FONT_HERSHEY_PLAIN, 5, (135,195,75), 8)
    cv2.putText(img, f"{down_count}", (1190,345), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255), 8)
    
    cv2.imshow('image', img)
    # cv2.imshow('imgRegion', imgRegion)
    cv2.waitKey(1)
    
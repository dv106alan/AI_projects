from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('./video/cars.mp4') # for video

model = YOLO('./yolo_weights/yolov8n.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

mask = cv2.imread('./image/cars-mask.png')

# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400,297,680,297]
total_count = 0
count_dict = {}

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    imgGraphic = cv2.imread('./image/graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphic, (0,0))
    
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
            
            if currentClass in ['car', 'truck', 'bus', 'motorbike'] and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detactions = np.vstack((detactions, currentArray))
                
    resultTracker = tracker.update(detactions)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)
    
    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        id = int(id)
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{id}', (x1, y1-20), 
                               scale=2, thickness=3, offset=10)
        
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx, cy), 5, (0,0,255), cv2.FILLED)
        
        id_index = f'{id}'
        if (limits[0]<cx<limits[2]) and (cy<limits[1]):
            count_dict[id_index] = 1
        elif limits[0]<cx<limits[2] and cy>limits[1] and id_index in count_dict:
            total_count += 1
            count_dict.pop(id_index)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)
    
    # cvzone.putTextRect(img, f'Count: {total_count}', (50, 50))
    cv2.putText(img, str(total_count), (235,100), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255), 8)
    
    cv2.imshow('image', img)
    # cv2.imshow('imgRegion', imgRegion)
    cv2.waitKey(1)
    
    
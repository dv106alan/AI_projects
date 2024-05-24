from ultralytics import YOLO
import cv2
import cvzone
import math
from PokerHandFunction import findPokerHand

cap = cv2.VideoCapture(0) # for webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('./yolo_weights/playingCards.pt')

classNames = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        hand_sets = set()
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))
            conf = math.ceil(box.conf[0]*100) / 100
            cls = int(box.cls[0])
            card_name = f"{classNames[cls]}"
            hand_sets.add(card_name)
            cvzone.putTextRect(img, f"{card_name} {conf}", (x1, y1-20), scale=1.2, thickness=2)
    
    if len(hand_sets) == 5:
        result_rank = findPokerHand(list(hand_sets))
        cvzone.putTextRect(img, f"Your hand: {result_rank}", (50, 50), scale=2, thickness=3)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)
import numpy as np
import cv2 as cv
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
from PIL import Image
import os

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

if(False and os.path.isfile('./yolov5s.pt')):
    model = torch.load('./yolov5s.pt')
else:
    print('Model not found in folder, downloading it from Torch Hub...')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        
    results = model(np.asanyarray(frame), size=640)

    transform = transforms.ToTensor()
    tens = transform(frame)

    res = draw_bounding_boxes(tens.mul(255).byte(), results.xyxy[0][:,:4])

    transform = transforms.ToPILImage()
    cv2_img = np.array(transform(res))
    
    for b in results.xyxy[0]:
        org = (int(b[0].item()), 
               int(b[1].item() + 20)
               )
        label = results.names[b[-1].item()]
        cv.putText(cv2_img, label, org, cv.FONT_HERSHEY_PLAIN, 2, (0,255,0))        

    cv.imshow('frame', cv2_img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

import numpy as np 
import cv2 as cv 

cap = cv.VideoCapture(r'video\demo_video.mp4')

if not cap.isOpened():
    print("unable to open your video file!") 
    exit() 

while True: 
    ret , frame = cap.read() 

    if not ret:
        break 
    cv.imshow('video from file',frame) 
    k = cv.waitKey(25)  ## 25 is best one delay to play a video
    if k == ord('q'):
        break 
cap.release() 
cv.destroyAllWindows()

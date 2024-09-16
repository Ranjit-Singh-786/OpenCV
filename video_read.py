import cv2 as cv 
import numpy as np 

cap = cv.VideoCapture(0)

if not cap.isOpened(): 
    print("not able to open camera!") 
    exit()

while True: 
    ret,frame = cap.read()
    # ret will have True if frame is captured otherwise False 
    # frame has single short image frame in the form of pixel
    if not ret:
        break
    cv.imshow("capturing your video ",frame)
    k = cv.waitKey(1)  # wait for 1 milisecond
    # k = cv.waitKey(0)  # wait for  indifinetly

    if k == ord('q'):
        break
cap.release()
cv.destroyAllWindows()     
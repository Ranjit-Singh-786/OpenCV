import cv2 as cv
import os 
os.makedirs('tested',exist_ok=True) 

img  = cv.imread('sadperson.jpeg')
cv.imshow("display image",img)

 

k = cv.waitKey(0)  # wait for indefinet  
if k == ord('s'):
    cv.imwrite('tested/research.jpg',img)
cv.destroyAllWindows()

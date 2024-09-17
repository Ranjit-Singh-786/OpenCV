import cv2 as cv 
import numpy as np 

# Load the image using OpenCV
img_arr = cv.imread('foto.jpg')
print(img_arr.shape)

# Extract the Red, Green, and Blue channels
blue_channel = img_arr[:, :, 0]
green_channel = img_arr[:, :, 1]
red_channel = img_arr[:, :, 2]

# Create blank images for each channel
zeros = np.zeros_like(blue_channel)

# Combine channels to show each color in its full color
blue_image = cv.merge([blue_channel, zeros, zeros])   # Blue channel image  
green_image = cv.merge([zeros, green_channel, zeros])  # Green channel image
red_image = cv.merge([zeros, zeros, red_channel])    # Red channel image





# Display the red, green, and blue images in color
cv.imshow("Red Channel (Red Image)", red_image)
k = cv.waitKey(0)
if k == ord('q'): 
    cv.destroyAllWindows()
    exit()

cv.imshow("Green Channel (Green Image)", green_image)
k = cv.waitKey(0)
if k == ord('q'):
    cv.destroyAllWindows()
    exit()

cv.imshow("Blue Channel (Blue Image)", blue_image)
cv.waitKey(0)
cv.destroyAllWindows()


# key points : before fetching the channels, image channels are in BGR form, actual form
# by default openCV load images in the form BGR form, you can convert it BGR to RGB 
# to convert BGR into RGB ==> img_rgb = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB) or img_rgb = img_bgr[:, :, ::-1]
# 0 ==> for blue
# 1 ==> for green 
# 2 ==> for red

# blue channel  ==> [blue channel , zero , zero]
# green channel ==> [zero , green , zero ] 
# red channel   ==> [zero,zero,red channel]
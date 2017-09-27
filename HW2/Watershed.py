'''
CSCI 677
Homework 2-b) Watershed Segmentor
Dixith Reddy Gomari
3098766483
gomari@usc.edu

References: Double click function:
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img3.jpg')
h,w,r=img.shape # Dimensions of the source image
markers = np.zeros((h,w), np.int32) # Markers image to use it in watershed with same size as the image

# Double click left button for marker location
print("Double click for giving the markers")
print("Press 'Esc' once done giving the marker loction")
def double_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
	markers[y][x] = np.random.randint(0,1000) # Storing a andom number for every marker location to assign a random color for segmentation
	print("Marker location: "+str(y)+","+str(x))

cv2.namedWindow('Image')
cv2.setMouseCallback('Image',double_click)

while(1):
    cv2.imshow('Image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

markers = cv2.watershed(img, markers)
# Assigns random colors to the markers for segmentation
for i in range(0,1000):
    img[markers == i] = [np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255)]

cv2.imshow('Segmented Image', img)
cv2.imwrite('Segmented_Image.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

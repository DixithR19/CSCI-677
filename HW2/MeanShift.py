'''
CSCI 677
Homework 2-a) Mean-Shift Segmentor
Dixith Reddy Gomari
3098766483
gomari@usc.edu

References:
http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
'''

import cv2

src = cv2.imread('img1.jpg')
dst = cv2.cvtColor(src, cv2.COLOR_BGR2LAB) # Conversion from RGB to LAB color space
cv2.pyrMeanShiftFiltering(dst,80,100,dst) # Vary sp,sr for variation in the segmented results


cv2.imshow('Image',src)
cv2.imshow('Segmented Image',dst)
cv2.imwrite('Segmented_Image.png',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

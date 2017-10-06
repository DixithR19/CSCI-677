'''
CSCI 677
Homework 3

Dixith Reddy Gomari
USC-ID:3098766483
gomari@usc.edu

References:
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
https://en.wikipedia.org/wiki/Random_sample_consensus
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('image_2.jpg') # queryImage
img2 = cv2.imread('image_5.jpg') # trainImage
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) # Covert to gray scale
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Draw Keypoints of eac image
cv2.drawKeypoints(gray1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Image1',img1)
cv2.imwrite('Image1.png',img1)
cv2.imwrite('Image2.png',img2)
cv2.imshow('Image2',img2)
print "No of Key features in Query Image: " +str(len(kp1))  #No of features in image 1
print "No of Key features in Train Image: " +str(len(kp2))  #No of features in image 2
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create Brute Force Matcher Object
bf=cv2.BFMatcher()

# Match Descriptors
matches = bf.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print "No of matching features between images for a lowe's ratio of 0.7: " +str(len(good))
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,              
           	   flags = 2)
# Draw Matches before RANSAC
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:20],None,**draw_params)

plt.imshow(img3),plt.show()


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)	# Applying RANSAC
    matchesMask = mask.ravel().tolist()
    print "Homography Matrix"
    print M
    h,w = gray1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask[:25], # draw only inliers
                   flags = 2)
# Draw Matches after RANSAC
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:25],None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()



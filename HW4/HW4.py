'''
Dixith Reddy Gomari
HW4-CSCI 677
USC-ID:3098766483
gomari@usc.edu

References:
https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
from mpl_toolkits.mplot3d import Axes3D

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# SIFT Matching

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

cv2.drawKeypoints(gray1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)

MIN_MATCH_COUNT = 10

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
		good.append(m)



# Essential Matrix

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

print good[0]
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:200],None,)

plt.imshow(img3),plt.show()

intrinsicMatrix = [[2760.0, 0.0, 	1520.0],
				   [0.0, 	2760.0, 1006.0],
				   [0.0, 	0.0,    1.0   ]]


i=np.array(intrinsicMatrix)

src_pts_l = cv2.undistortPoints(src_pts, np.array(intrinsicMatrix), distCoeffs=None)
dst_pts_r = cv2.undistortPoints(dst_pts, np.array(intrinsicMatrix), distCoeffs=None)

E, mask = cv2.findEssentialMat(src_pts, dst_pts, i,method=cv2.RANSAC) 
points, R, t, mask=cv2.recoverPose(E, src_pts, dst_pts)
print E
print 'Rotation Matrix'
print R
print 'Translation Matrix'
print t


# Triangulate points

M_r = np.hstack((R,t))
M_l = np.hstack((np.eye(3,3), np.zeros((3,1))))

P_l = np.dot(intrinsicMatrix,  M_l)
print 'Camera Matrix for the first camera'
print P_l
P_r = np.dot(intrinsicMatrix,  M_r)
print 'Camera Matrix for the Second camera'
print P_r

point_4d_hom = cv2.triangulatePoints(P_l, P_r, src_pts_l, dst_pts_r)
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_4d = point_4d[:3, :].T


# Visulaization
fig=plt.figure()
ax=Axes3D(fig)

ax.scatter(point_4d[:,0],point_4d[:,1],point_4d[:,2])
plt.show()

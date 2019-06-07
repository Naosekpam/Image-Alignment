import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randrange

img_ = cv2.imread('C:/Users/veronica/OpenCV_17d/1.jpg')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread('C:/Users/veronica/OpenCV_17d/2.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
# BFMatcher with default params
# FLANN parameters
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
 # Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

## Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.5*m[1].distance:         
     	good.append(m)
matches1 = np.asarray(good)
 	 
 # ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
	if m.distance < 0.7*n.distance:
		matchesMask[i]=[1,0]

if len(matches1[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches1[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches1[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
else:
    raise AssertionError("Can't find enough keypoints.")  

draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow("correspondences", img3)
cv2.waitKey()

dst = cv2.warpPerspective(img_,H,(img.shape[1]+img_.shape[1], img_.shape[0]))     	
plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()





import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def image_hist(image):
    color = ('b', 'g', 'r')   #
    for i , color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])  #
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()

img = cv2.imread("./571.png")
#img = cv2.imread("./1.jpg")
#plot_demo(img)
#image_hist(img)

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8  # classify color
ret,label,center=cv2.kmeans(Z,K,None,criteria,15,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

ret, res3_bone = cv2.threshold(img,220,225,cv2.THRESH_BINARY)
res4_boneReduce = (res3_bone-res2)

#select the specific region you want to mask
ret, res4_boneReduce_threshold = cv2.threshold(res4_boneReduce,115,0,cv2.THRESH_TOZERO_INV)
ret, res4_boneReduce_threshold = cv2.threshold(res4_boneReduce_threshold,65,255,cv2.THRESH_BINARY)
cv2.imshow("x",res4_boneReduce_threshold)
cv2.waitKey(0)

#copy the data
res4_temp = np.zeros(res4_boneReduce_threshold.shape,np.uint8)
res4_temp = res4_boneReduce_threshold.copy()
gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
#copy the gray image because function
gray_temp = gray.copy()
#findContours will change the imput image into another

ret, contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#show the contours of the imput image
cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)

#find the max area of all the contours and fill it with 0
area = []

for i in range(len(contours)):
    area.append(cv2.contourArea(contours[i]))

max_idx = np.argmax(area)
cv2.fillPoly(gray, contours[max_idx], 0)
#cv2.fillConvexPoly(gray,contours[max_idx,0])
#test = cv2.fillConvexPoly(gray, contours[max_idx], 0)
#cv2.imshow("1", test)
res4_boneReduce_threshold = cv2.cvtColor(res4_boneReduce_threshold, cv2.COLOR_BGR2GRAY)
out_put = ( res4_boneReduce_threshold - gray)

#resr_boneReduce_threshold (512 512 3)
#gray (512 512)

th, im_th = cv2.threshold(out_put, 220, 255, cv2.THRESH_BINARY_INV);

h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(out_put, mask, (0,0), 255)
out_put_liverMask = cv2.bitwise_not(out_put)

img_512_512_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Result = out_put|img_512_512_1

cv2.imshow("res_origin",img)
cv2.imshow('res_Kmeans',res2)
cv2.imshow("res_Bin_threshold", res3_bone)
cv2.imshow("res_Reduce_Bonre",res4_boneReduce)
cv2.imshow("res4_boneReduce_threshold",res4_boneReduce_threshold)
cv2.imshow("res4_temp",res4_temp)
cv2.imshow("gray",gray)
cv2.imshow("liver_Inv",out_put)
cv2.imshow("out_put_liverMask",out_put_liverMask)
cv2.imshow("Result",Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import os

#https://www.programcreek.com/python/example/70455/cv2.drawContours

def Kmean(img):

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res_center = center[label.flatten()]
    res2_output = res_center.reshape((img.shape))

    # ret, res3_bone = cv2.threshold(img,100,225,cv2.THRESH_BINARY)

    # res4_boneReduce = (res3_bone-res2)

    # select the specific region you want to mask
    # ret, res4_boneReduce_threshold = cv2.threshold(res2,150,0,cv2.THRESH_TOZERO_INV)
    cv2.imshow("res2", res2_output)
    cv2.waitKey(0)
    return res2_output

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    #input, gives all the contours, contour approximation compresses horizontal,
    #vertical, and diagonal segments and leaves only their end points. For example,
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology.
    #It has as many elements as the number of contours.
    #we dont need it
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

#109 425
#117 418
patient_num = 115
i = 568
def file_name(file_dir):
    L = []
    numofFile = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                numofFile += 1
                L.append(os.path.join(root, file))
                #print (file)
    return L,numofFile

_, numberofFile = file_name("../result/"+str(patient_num)+"Process/")
print (numberofFile)


res = cv2.imread("../Mat_pic/"+str(patient_num)+"/horizontal_plane/"+str(i)+".png")
res_copy = cv2.imread("../result/"+str(patient_num)+"Process/"+str(i)+".png")

mask_tf = cv2.imread("../result/"+str(patient_num)+"Process/"+str(i)+".png")
mask_tf = rotate(mask_tf, 90)
#mask = rotate(mask,90)
Image = mask_tf & res
cv2.imshow("map",Image)

#cv2.imshow("568",Image)
#cv2.waitKey(0)
res = Image

img = mask_tf & res
res = Kmean(img)

imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 150, 255, 0)
#thresh = imgray
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#V1
# biggest, imag = find_biggest_contour(thresh)
# cv2.imshow("Biggest_counter",imag)
# cv2.waitKey(0)

#V2
# area = []
# for i in range(len(contours)):
#     area.append(cv2.contourArea(contours[i]))
# max_idx = np.argmax(area)
# print contours[max_idx]
# cv2.fillPoly(image, contours[max_idx], 0)
#
# #imag = cv2.drawContours(res,contours[max_idx],-1,(0,0,0),-1)
# cv2.imshow("V1", image)
# cv2.waitKey(0)

#V3
c_max = []
max_area = 0
max_cnt = 0
for x in range(len(contours)):
    cnt = contours[x]
    area = cv2.contourArea(cnt)
    # find max countour
    if (area > max_area):
        if (max_area != 0):
            c_min = []
            c_min.append(max_cnt)
            cv2.drawContours(res, c_min, -1, (255, 0, 0), cv2.FILLED)
        max_area = area
        max_cnt = cnt
    else:
        c_min = []
        c_min.append(cnt)
        cv2.drawContours(res, c_min, -1, (255, 0, 0), cv2.FILLED)

c_max.append(max_cnt)
# cv2.fillPoly(res, c_max,[255,255,255])
# cv2.imshow("fillPoly",res)
# cv2.waitKey(0)

imag_green = cv2.drawContours(res, c_max, -1, (255, 0, 0), -1)
cv2.imshow("x", imag_green)
cv2.waitKey(0)

imag = cv2.drawContours(res, c_max, -1, (255, 255, 255), -1)
th, im_th = cv2.threshold(imag, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("th", im_th)
cv2.waitKey(0)

kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(im_th, -1, kernel)
result=cv2.GaussianBlur(im_th, (11, 11), 4)
th, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
result = rotate(result, 270)

#cv2.imwrite("../result/"+str(patient_num)+"ProcessMask/"+str(i)+".png",result)

cv2.imshow("result :"+str(i),result)
#cv2.imwrite("D:\\毕业设计文章\\研究报告\\图片\\GaussianBlur\\map.png",result)
cv2.waitKey(0)

res = cv2.imread("../Mat_pic/"+str(patient_num)+"/horizontal_plane/"+str(i)+".png")

mask_tf = result
mask_tf = rotate(mask_tf, 90)
#mask = rotate(mask,90)
Image = mask_tf & res
cv2.imshow("map result",Image)

cv2.waitKey(0)


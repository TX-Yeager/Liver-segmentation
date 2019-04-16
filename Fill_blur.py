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

def optimizer_CT_Blur(path):
    img = cv2.imread(path)

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    img = cv2.blur(gradient, (3, 3))

    # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    ret, res4_boneReduce_threshold = cv2.threshold(res2, 150, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow("x", res2)
    cv2.waitKey(0)

    # copy the data
    res4_temp = np.zeros(res4_boneReduce_threshold.shape, np.uint8)
    res4_temp = res4_boneReduce_threshold.copy()
    gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
    # copy the gray image because function
    gray_temp = gray.copy()
    # findContours will change the imput image into another

    ret, contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # #show the contours of the imput image
    # cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)
    #
    # #find the max area of all the contours and fill it with 0
    # area = []
    #
    # for i in range(len(contours)):
    #     area.append(cv2.contourArea(contours[i]))
    #
    # max_idx = np.argmax(area)
    # cv2.fillPoly(gray, contours[max_idx], 0)
    # cv2.imshow("fill",gray)
    # cv2.waitKey(0)

    c_max = []
    max_area = 0
    max_cnt = 0
    for x in range(len(contours)):
        cnt = contours[x]
        area = cv2.contourArea(cnt)
        # find max countour
        if (area >= max_area):
            if (max_area != 0):
                c_min = []
                c_min.append(max_cnt)
                cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)
            max_area = area
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)

    c_max.append(max_cnt)

    imag = cv2.drawContours(gray_temp, c_max, -1, (255, 255, 255), cv2.FILLED)

    th, im_th = cv2.threshold(imag, 200, 255, cv2.THRESH_BINARY)
    output_mask = im_th
    return output_mask


def optimizer_CT(path):
    img = cv2.imread(path)


    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # ret, res3_bone = cv2.threshold(img,100,225,cv2.THRESH_BINARY)

    # res4_boneReduce = (res3_bone-res2)

    # select the specific region you want to mask
    ret, res4_boneReduce_threshold = cv2.threshold(res2, 150, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow("x", res2)
    cv2.waitKey(0)

    # copy the data
    res4_temp = np.zeros(res4_boneReduce_threshold.shape, np.uint8)
    res4_temp = res4_boneReduce_threshold.copy()
    gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
    # copy the gray image because function
    gray_temp = gray.copy()
    # findContours will change the imput image into another

    ret, contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # show the contours of the imput image
    cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)

    # find the max area of all the contours and fill it with 0
    area = []

    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(area)
    cv2.fillPoly(gray, contours[max_idx], 0)

    # cv2.fillConvexPoly(gray,contours[max_idx,0])
    # test = cv2.fillConvexPoly(gray, contours[max_idx], 0)
    # cv2.imshow("1", test)
    res4_boneReduce_threshold = cv2.cvtColor(res4_boneReduce_threshold, cv2.COLOR_BGR2GRAY)
    out_put = (res4_boneReduce_threshold - gray)

    # resr_boneReduce_threshold (512 512 3)
    # gray (512 512)

    th, im_th = cv2.threshold(out_put, 200, 255, cv2.THRESH_BINARY_INV);

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(out_put, mask, (0, 0), 255)
    out_put_liverMask = cv2.bitwise_not(out_put)
    return out_put_liverMask




img = cv2.imread("./568.png")
#img = cv2.imread("./1.jpg")
#plot_demo(img)
#image_hist(img)

# gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
#
# gradient = cv2.subtract(gradX, gradY)
# gradient = cv2.convertScaleAbs(gradient)
#
# img = cv2.blur(gradient, (3, 3))

img = cv2.GaussianBlur(img, (11, 11), 1)
#img = cv2.medianBlur(img, 5)
#(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # classify color
ret,label,center=cv2.kmeans(Z,K,None,criteria,15,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#ret, res3_bone = cv2.threshold(img,100,225,cv2.THRESH_BINARY)

#res4_boneReduce = (res3_bone-res2)


#select the specific region you want to mask
ret, res4_boneReduce_threshold = cv2.threshold(res2,150,0,cv2.THRESH_TOZERO_INV)
cv2.imshow("x",res2)
cv2.waitKey(0)

#copy the data
res4_temp = np.zeros(res4_boneReduce_threshold.shape,np.uint8)
res4_temp = res4_boneReduce_threshold.copy()
gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
#copy the gray image because function
gray_temp = gray.copy()
#findContours will change the imput image into another

contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# #show the contours of the imput image
# cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)
#
# #find the max area of all the contours and fill it with 0
# area = []
#
# for i in range(len(contours)):
#     area.append(cv2.contourArea(contours[i]))
#
# max_idx = np.argmax(area)
# cv2.fillPoly(gray, contours[max_idx], 0)
# cv2.imshow("fill",gray)
# cv2.waitKey(0)

c_max = []
max_area = 0
max_cnt = 0
for x in range(len(contours)):
    cnt = contours[x]
    area = cv2.contourArea(cnt)
    # find max countour
    if (area >= max_area):
        if (max_area != 0):
            c_min = []
            c_min.append(max_cnt)
            cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)
        max_area = area
        max_cnt = cnt
    else:
        c_min = []
        c_min.append(cnt)
        cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)

c_max.append(max_cnt)

imag = cv2.drawContours(gray_temp, c_max, -1, (255, 255, 255), cv2.FILLED)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
# imag = cv2.morphologyEx(imag, cv2.MORPH_CLOSE, kernel)

#closed = cv2.erode(closed, None, iterations=4)
#closed = cv2.dilate(closed, None, iterations=4)

th, im_th = cv2.threshold(imag, 200, 255, cv2.THRESH_BINARY)
output_mask = im_th
cv2.imshow("fill",output_mask)
cv2.waitKey(0)

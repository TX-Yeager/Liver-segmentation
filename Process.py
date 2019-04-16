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

def optimizer_CT(path):
    img = cv2.imread(path)
    # img = cv2.imread("./1.jpg")
    # plot_demo(img)
    # image_hist(img)

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
    # cv2.imshow("x", res2)
    # cv2.waitKey(0)

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


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def contours_process(res_path,mask_path):

    res = cv2.imread(res_path)
    mask_tf = cv2.imread(mask_path)
    mask_tf = rotate(mask_tf, 90)
    # mask = rotate(mask,90)
    Image = mask_tf & res

    res = Image
    cv2.imshow("resource", res)
    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c_max = []
    max_area = 0
    max_cnt = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # find max countour
        if (area > max_area):
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
    imag = cv2.drawContours(res, c_max, -1, (255, 255, 255), -1)
    th, im_th = cv2.threshold(imag, 250, 255, cv2.THRESH_BINARY);
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(im_th, -1, kernel)
    result = cv2.blur(im_th, (5, 5))
    th, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY);
    result = rotate(result, 270)
    return result

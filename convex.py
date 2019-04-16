import numpy as np
import cv2
res = cv2.imread("chanv_test.png")
cv2.imshow("resource",res)
cv2.waitKey(0)
imgray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,150,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# print(hierarchy)
# area = []
# for i in range(len(contours)):
#     area.append(cv2.contourArea(contours[i]))
# max_idx = np.argmax(area)
# print contours[max_idx]
# imag = cv2.drawContours(res,contours[max_idx],-1,(0,255,0),-1)

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
imag = cv2.drawContours(res,c_max,-1,(0,255,0),-1)
th, im_th = cv2.threshold(imag, 250, 255, cv2.THRESH_BINARY)

cv2.imshow("image", imag)
cv2.waitKey(0)
# th, im_th = cv2.threshold(imag, 128, 255, cv2.THRESH_BINARY);

kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(im_th, -1, kernel)
result=cv2.blur(im_th, (5, 5))
th, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
result = rotate(result,270)

cv2.imshow("result",result)
#cv2.imwrite("contours_result_573.png",result)
cv2.waitKey(0)

#hull = cv2.convexHull(points,hull,clockwise,returnPoints)
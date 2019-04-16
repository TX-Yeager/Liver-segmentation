import cv2


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
mask = cv2.imread("/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database/liver_seg/115/576.png")

res = cv2.imread("./horizontal_plane/576.png")
mask_tf = cv2.imread("./576mask.png")
mask_tf = rotate(mask_tf, 90)
mask = rotate(mask,90)

cv2.waitKey(0)
result_tf = res & mask_tf
result = res & mask
cv2.imshow("result_tf",result_tf)
cv2.imshow("result",result)
cv2.waitKey(0)

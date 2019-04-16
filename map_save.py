import cv2

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
#mask = cv2.imread("/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database/liver_seg/115/573.png")

res = cv2.imread("./horizontal_plane/573.png")
mask_tf = cv2.imread("./573mask.png")
mask_tf = rotate(mask_tf, 90)
#mask = rotate(mask,90)
Image = mask_tf & res
cv2.imwrite("chanv_test.png",Image)
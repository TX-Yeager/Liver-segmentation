import cv2
patient_num = 115

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
patient_num = 115
i = 568
res = cv2.imread("../../Mat_pic/" + str(patient_num) + "/horizontal_plane/" + str(i) + ".png")
res_copy = cv2.imread("../result/" + str(patient_num) + "ProcessMask/" + str(i) + ".png")

mask_tf = cv2.imread("../result/" + str(patient_num) + "Process/" + str(i) + ".png")
mask_tf = rotate(mask_tf, 90)
# mask = rotate(mask,90)
Image = mask_tf & res
Path = "../result/" + str(patient_num) + "Map/" + str(i) + ".png"
Path ="/home/chacky/PycharmProjects/untitled3/Doc_week5/"+str(patient_num)+"_"+str(i)+"ProcessV1Mask.png"
cv2.imwrite(Path, Image)
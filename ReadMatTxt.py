import os
import pickle
import numpy as np
import cv2

def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0)
    return new_img

num_patient = 15

mat_txt = open("./Mat2Txt/Matrix_"+str(num_patient)+".txt",mode="rb")
cube = pickle.load(mat_txt)
shape = cube.shape
print (shape)
print (shape[1])

for i in range(cube.shape[0]):
    img = cube[i,:,:]
    img = RotateAntiClockWise90(img)
    cv2.imshow("pic",img)
    key=cv2.waitKey(30)
    if(key == 27):
        break

for i in range(cube.shape[1]):
    img = cube[:, i, :]
    img = RotateAntiClockWise90(img)
    cv2.imshow("pic",img)
    key=cv2.waitKey(30)
    if(key == 27):
        break

for i in range(cube.shape[2]):
    img = cube[:, :, i]
    img = RotateAntiClockWise90(img)
    img = RotateAntiClockWise90(img)
    cv2.imshow("pic",img)
    key=cv2.waitKey(30)
    if(key == 27):
        break
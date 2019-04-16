import os
import pickle
import numpy as np
import cv2
#Image Rotate
def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0 )
    return new_img
#Movie True of False
movie = True
#Load Matrix data
num_patient = 115
mat_txt = open("./Mat2Txt/Matrix_"+str(num_patient)+".txt",mode="rb")
cube = pickle.load(mat_txt)
#Make Diraction
path_patient = "./"+str(num_patient)
path_patient_horizontal = "./" + str(num_patient) + "/horizontal_plane/"
path_patient_sagittal = "./" + str(num_patient) + "/sagittal_plane/"
path_patient_coronal = "./" + str(num_patient) + "/coronal_plane/"
if not (os.path.exists(path_patient) and \
        os.path.exists(path_patient_coronal)and\
        os.path.exists(path_patient_horizontal) and\
        os.path.exists(path_patient_sagittal)):
    os.mkdir("./"+str(num_patient))
    os.mkdir("./"+str(num_patient)+"/horizontal_plane/")
    os.mkdir("./"+str(num_patient)+"/sagittal_plane/")
    os.mkdir("./"+str(num_patient)+"/coronal_plane/")
    print ("Make New Path")

#save png
for i in range(cube.shape[0]):
    img = cube[i,:,:]
    img = RotateAntiClockWise90(img)
    if movie == True:
        cv2.imshow("pic",img)
    cv2.imwrite("./"+str(num_patient)+"/horizontal_plane/"+str(i)+".png",img)
    print (i)
    key=cv2.waitKey(1)
    if(key == 27):
        break

for i in range(cube.shape[1]):
    img = cube[:, i, :]
    img = RotateAntiClockWise90(img)
    if movie == True:
        cv2.imshow("pic",img)
    cv2.imwrite("./"+str(num_patient)+"/sagittal_plane/"+str(i)+".png",img)
    print (i)
    key=cv2.waitKey(1)
    if(key == 27):
        break

for i in range(cube.shape[2]):
    img = cube[:, :, i]
    img = RotateAntiClockWise90(img)
    img = RotateAntiClockWise90(img)
    if movie == True:
        cv2.imshow("pic",img)
    cv2.imwrite("./"+str(num_patient)+"/coronal_plane/"+str(i)+".png",img)
    print (i)
    key=cv2.waitKey(1)
    if(key == 27):
        break
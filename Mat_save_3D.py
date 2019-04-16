from scipy.io import loadmat
import cv2
import numpy as np
num_m = 1
num_patient = 117
import os
import pickle
#Good Performance patient 107  pic:486
np.set_printoptions(threshold=np.inf)
mat_txt = open("mat.txt",mode="wb")
def file_name(file_dir):
    L = []
    numofFile = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mat':
                numofFile += 1
                L.append(os.path.join(root, file))
                print (file)
    return L,numofFile
path = "./LiTS_database/images_volumes/"+str(num_patient)+"/"
L, num_File = file_name(path)
print (num_File)
cube = np.zeros(shape = (num_File,512,512),dtype='uint8')
count = 0
for i in range(num_File):
    m = loadmat(path+str(i+1)+".mat")
    matrix = np.array(m['section'],dtype="uint8")
    cube[i] = matrix
print (cube)
print (cube.shape)
print (cube.dtype)

pickle.dump(cube, mat_txt)
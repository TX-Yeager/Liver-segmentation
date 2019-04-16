import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_dice_CT_pic(output_path,label_path):
    matrix_label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
    matrix_label = matrix_label > 128
    # matrix_label = np.expand_dims(matrix_label, axis=0)
    matrix_label = matrix_label.astype(np.float32)

    matrix_output = cv2.imread(output_path ,cv2.IMREAD_GRAYSCALE)
    matrix_output = matrix_output > 128
    #cv2.imshow("CT",matrix_output.astype(np.uint8))
    #cv2.waitKey(0)
    # matrix_output = np.expand_dims(matrix_output, axis=0)
    matrix_output = matrix_output.astype(np.float32)

    print(matrix_label.sum(), matrix_output.sum())
    return matrix_output,matrix_label


def next_pic(output_dir , label_dir, number_CT):
    Outpath = output_dir + str(number_CT)+".png"
    Labelpath = label_dir + str(number_CT) + ".png"
    return Outpath,Labelpath


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


patient_num = 11
label_dir = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database/liver_seg/" + str(
    patient_num) + "/"
output_dir_r = "../result/" + str(patient_num) + "/"
_, numberofFile = file_name("../result/" + str(patient_num) + "/")
print (numberofFile)
loss_sum = 0
path = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/statistics/"+str(patient_num)+"Statistics/"
# if not os.path.exists(path):
#     os.mkdir(path)
y = []
x = []
for i in range(1, numberofFile):
    print("CT num is :" + str(i))
    output_path, label_path = next_pic(output_dir_r, label_dir, i)
    matrix_output, matrix_label = prepare_dice_CT_pic(output_path, label_path)
    y.append(matrix_label.sum())
    x.append(i)

plt.plot(x, y, 'r', lw=1, marker='.')
plt.show()



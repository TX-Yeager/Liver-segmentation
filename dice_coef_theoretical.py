import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
from tensorflow.contrib.layers.python.layers import utils
import scipy
from dataset.dataset_seg import Dataset
import cv2
from scipy.io import loadmat
##Question! Patient 115 : 567->582 black hole in the liver!
##Probability has some problems.
#import matplotlib
import matplotlib.pyplot as plt
import filling_hole  as fillHole

def dice_coef_theoretical(y_pred, y_true):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
        """

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

    y_pred_f = tf.nn.sigmoid(y_pred)
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)
    #if the pictures' pixel is zero that means that is nothing masked in the ct
    # so the dice is 1.
    y_pred_sum = tf.reduce_sum(y_pred)
    y_true_sum = tf.reduce_sum(y_true)
    # if (y_pred_sum == 0) and (y_true_sum == 0):
    #     dice = 1

    def pred_f1(): return 0.0

    def pred_f2(): return y_pred_sum
    y_pred_sum= tf.cond(tf.less(y_pred_sum, 500), pred_f1, pred_f2)

    def f0():
        result = tf.cond(tf.equal(y_true_sum, 0), f1, f2)
        return result

    def f1(): return 1.0

    def f2(): return dice

    result = tf.cond(tf.equal(y_pred_sum, y_true_sum), f0, f2)
    dice = result
    return dice, y_pred_sum, y_true_sum

def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss

    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)

    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = 0.931 * loss_pos + 0.069 * loss_neg

    return final_loss

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

ones_matrix = np.ones((512,512),dtype=np.float32)
ones_matrix = np.zeros((512,512),dtype=np.float32)
label_path = "./liver_seg/115/575.png"
#label_path = "./result/115/533.png"
#output_path = "./result/115/575mask.png"
output_path = "./result/115/575.png"
#output_path = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database/liver_seg/115/512.png"
matrix_label,matrix_output = prepare_dice_CT_pic(output_path, label_path)
matrix_label_ones = tf.Variable(ones_matrix)
matrix_output_ones = tf.Variable(ones_matrix)



tf_matrix_label = tf.placeholder(tf.float32, [512,512])
tf_matrix_output = tf.placeholder(tf.float32, [512,512])

dice, y_pred_sum,y_true_sum = dice_coef_theoretical(tf_matrix_output, tf_matrix_label)
dice_matrix = dice_coef_theoretical(matrix_output_ones, matrix_label_ones)
loss = class_balanced_cross_entropy_loss(tf_matrix_output, tf_matrix_label)

y1 = []
y2 = []
x = []
mean = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run([dice,y_pred_sum,y_true_sum], feed_dict={tf_matrix_label:matrix_label, tf_matrix_output:matrix_output})
    # res_matrix= sess.run(dice_matrix)
    # loss_CT= sess.run(loss,feed_dict={tf_matrix_label:matrix_label, tf_matrix_output:matrix_output})

    print ("CT compare", res[0])
    #print ("Loss CT", loss_CT)
    #print ("Same matrix compare",res_matrix)
    #res_np = res.astype(np.float32)[0, :, :, 3 / 2]


    sum = 0
    sum1 = 0
    patient_num = 115
    output_dir_r = "./result/"+str(patient_num)+"/"
    #output_dir_P = "./result/" + str(patient_num) + "Process/"
    #output_dir_P = "./result/"+str(patient_num)+"ProcessMask/"
    #output_dir_P = "./result/" + str(patient_num) + "DensecrfMask/"
    output_dir_P = "./result/" + str(patient_num) + "DensecrfMask_blur/"

    label_dir = "./liver_seg/"+str(patient_num)+"/"
    _, numberofFile = file_name("./result/"+str(patient_num)+"/")
    print (numberofFile)
    loss_sum = 0
    for i in range(1, numberofFile):
    #for i in range(413, 598):
        output_path, label_path = next_pic(output_dir_r, label_dir, i)
        output_path_P, label_path = next_pic(output_dir_P, label_dir, i)

        matrix_output, matrix_label = prepare_dice_CT_pic(output_path,label_path)
        matrix_output_P, matrix_label = prepare_dice_CT_pic(output_path_P,label_path)

        #label
        res = sess.run([dice,y_pred_sum,y_true_sum],
                       feed_dict={tf_matrix_label: matrix_label, tf_matrix_output: matrix_output})
        print ("CT "+str(i)+" compare", res[0])
        y_2 = res[0]
        entropy = sess.run(loss,
                           feed_dict={tf_matrix_label: matrix_label, tf_matrix_output: matrix_output})
        loss_sum += entropy

        #processMask
        res = sess.run([dice, y_pred_sum, y_true_sum],
                       feed_dict={tf_matrix_label: matrix_label, tf_matrix_output: matrix_output_P})
        print ("CT " + str(i) + " Process", res[0])
        if(res[0]<0.9):
            print ("patien number of :"+str(i)+" ,Pix ="+str(matrix_output.sum()))
            #19000
        sum += res[0]
        sum1 += y_2
        mean.append(matrix_label.sum())
        x.append(i)
        y1.append(res[0])
        y2.append(y_2)
    #numberofFile = 185
    print ("V1",sum1/numberofFile)
    print ("V2",sum/numberofFile)
    print("Loss",loss_sum/numberofFile)
    import matplotlib
    myfont = matplotlib.font_manager.FontProperties(fname="simsun.ttc")
    fig = plt.figure()

    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(1, 1, 1)
    # ax1.plot(x, y1, 'r', lw=1, marker='.', label="CRF")
    ax1.plot(x, y2, 'b', lw=1,  marker='',label="Only FCN")
    # ax1.legend(loc=1)
    # # p2 = ax1.plot([3, 2, 1], label="Only FCN")
    # # p1 = ax1.plot([1, 2, 3], label="After Process")
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles[::-1], labels[::-1])
    # ax1.set_ylabel('Similarity')
    # ax1.set_xlabel('CT Number')
    # plt.grid()
    #
    # ax2 = ax1.twinx()
    # ax2.plot(x, mean, 'g', lw=1, marker='.', label="Liver Size")
    # ax2.legend(loc=1)
    # ax2.set_ylabel('Liver Segmentation Size')
    # #ax2.set_title("Compare")
    # plt.grid()

    # plt.figure(figsize=(20, 10))
    # plt.plot(x, mean/np.max(mean), 'g', lw=1, marker='.')
    # plt.plot(x, y1, 'r', lw=1,marker='.')
    # plt.plot(x, y2, 'b', lw=1)
    # #plt.savefig('/home/chacky/PycharmProjects/untitled3/Doc/Patient_Blur_MaskProcess_LiverPosition' + str(patient_num) + '.jpg')
    plt.show()

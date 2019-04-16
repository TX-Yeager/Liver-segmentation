import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(gpu_options=gpu_options)
from tensorflow.contrib.layers.python.layers import utils
import os

file = open("Tensor_Net", 'rb')
result = pickle.load(file)
matrix_res = cv2.imread("467.png")
matrix = np.expand_dims(matrix_res, axis=0)
matrix = matrix.astype(np.float32)
input_label = tf.Variable(matrix)
input_image = result
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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    final_loss = class_balanced_cross_entropy_loss(input_image, input_label)
    loss = sess.run(final_loss)
    print (loss)

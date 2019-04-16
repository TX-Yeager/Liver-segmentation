import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(gpu_options=gpu_options)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

matrix_res = cv2.imread("345.png")
matrix = np.expand_dims(matrix_res, axis=0)
matrix = matrix.astype(np.float32)
input_image = tf.Variable(matrix)
# input = tf.placeholder(tf.float32, [1, 512, 512, 3])

print ("Resource 4-D tensor", matrix.shape)
# convolution img

number_slices = 3


def conv2dcut(input_image):
    net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
    net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
    net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
    net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    print (net_2.shape)
    side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
    side_2_s = slim.conv2d(side_2, number_slices, [1, 1], scope='score-dsn_2')


    #cut convolutiona
    final_layer = slim.conv2d(side_2_s, 3, [1, 1], scope="cut")
    return final_layer


output = conv2dcut(input_image)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    slim_max_pool2d_value = sess.run(output)
    print ("Result 4-D tensor", slim_max_pool2d_value.shape)
    result = slim_max_pool2d_value
    depth = result.shape[3]
    cv2.imshow("res_pic", matrix_res)
    for i in range(depth):
        cv2.imshow("max_pool2d", result[0, :, :, i])
        cv2.waitKey(0)
    cv2.waitKey(0)
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())


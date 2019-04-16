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

def VGG16(input_image):
    input_1 = slim.conv2d(input_image, 64, [3, 3], scope="conv1_1")
    input_1 = slim.conv2d(input_1, 64, [3, 3], scope="conv1_2")
    # pooling img
    pool1 = slim.max_pool2d(input_1, [2, 2], scope='pool1')

    input_2 = slim.conv2d(pool1, 128, [3, 3], scope="conv2_1")
    input_2 = slim.conv2d(input_2, 128, [3, 3], scope="conv2_2")
    # pooling img
    pool2 = slim.max_pool2d(input_2, [2, 2], scope='pool2')

    input_3 = slim.conv2d(pool2, 256, [3, 3], scope="conv3_1")
    input_3 = slim.conv2d(input_3, 256, [3, 3], scope="conv3_2")
    input_3 = slim.conv2d(input_3, 256, [3, 3], scope="conv3_3")
    # pooling img
    pool3 = slim.max_pool2d(input_3, [2, 2], scope='pool3')

    input_4 = slim.conv2d(pool3, 512, [3, 3], scope="conv4_1")
    input_4 = slim.conv2d(input_4, 512, [3, 3], scope="conv4_2")
    input_4 = slim.conv2d(input_4, 512, [3, 3], scope="conv4_3")
    # pooling img
    pool4 = slim.max_pool2d(input_4, [2, 2], scope='pool4')

    input_5 = slim.conv2d(pool4, 512, [3, 3], scope="conv5_1")
    input_5 = slim.conv2d(input_5, 512, [3, 3], scope="conv5_2")
    input_5 = slim.conv2d(input_5, 512, [3, 3], scope="conv5_3")
    # pooling img
    pool5 = slim.max_pool2d(input_5, [2, 2], scope='pool5')

    net = slim.flatten(pool5)

    fc1 = slim.fully_connected(pool5, 4096, scope='fc1')
    fc2 = slim.fully_connected(fc1, 4096, scope='fc2')
    fc3 = slim.fully_connected(fc2, 1000, activation_fn=None, scope='fc3')

    final_layer = fc3
    return final_layer


output = VGG16(input_image)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    slim_max_pool2d_value = sess.run(output)
    print ("Result 4-D tensor", slim_max_pool2d_value.shape)
    result = slim_max_pool2d_value
    depth = result.shape[3]
    cv2.imshow("res_pic", matrix_res)
    for i in range(depth):
        cv2.imshow("max_pool2d", result[0, :, :, i])
        cv2.waitKey(10)
    cv2.waitKey(0)
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())


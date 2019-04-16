import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

matrix_res = cv2.imread("345.png")
matrix = np.expand_dims(matrix_res, axis=0)
matrix = matrix.astype(np.float32)
input_image = tf.Variable(matrix)
# input = tf.placeholder(tf.float32, [1, 512, 512, 3])
number_slices = 3
print ("Resource 4-D tensor", matrix.shape)
im_size = tf.shape(input_image)


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


# convolution img
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
side_2_s_t = slim.convolution2d_transpose(side_2_s, number_slices, 4, 2, scope='score-dsn_2-up')
side_2_s_crop = crop_features(side_2_s_t, im_size)
side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
side_2_f = crop_features(side_2_f, im_size)

#conv2d_transpose = slim.convolution2d_transpose(net_2, 512, 2, 1, scope='score-dsn_5_1-up')

final_layer = side_2_f

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    slim_max_pool2d_value = sess.run([final_layer, side_2_s_t, side_2_s, side_2, side_2_f])
    print ("Result 4-D tensor", slim_max_pool2d_value[0].shape)
    print ("side_2", slim_max_pool2d_value[3].shape)
    print ("side_2_s", slim_max_pool2d_value[2].shape)
    print ("side_2_s_t", slim_max_pool2d_value[1].shape)
    print ("side_2_s_f", slim_max_pool2d_value[4].shape)

    result = slim_max_pool2d_value[0]
    depth = result.shape[3]
    cv2.imshow("res_pic", matrix_res)
    for i in range(depth):
        cv2.imshow("max_pool2d", result[0, :, :, i])
        cv2.waitKey(0)
    cv2.waitKey(0)
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())


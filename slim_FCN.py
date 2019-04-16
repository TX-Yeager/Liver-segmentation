import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
from tensorflow.contrib.layers.python.layers import utils
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
#matrix_res = cv2.imread("im1.png")
#matrix_res = cv2.imread("467.png")
matrix_res = cv2.imread("467source.png")
#matrix_res = cv2.imread("./tensorflow-fcn-master/test_data/tabby_cat.png")
matrix = np.expand_dims(matrix_res, axis=0)
matrix = matrix.astype(np.float32)
input_image = tf.Variable(matrix)

# input = tf.placeholder(tf.float32, [1, 512, 512, 3])

print ("Resource 4-D tensor", matrix.shape)


def load_vgg_imagenet(ckpt_path, number_slices):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes t1he network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            if not "conv1/conv1_1/weights" in v or number_slices < 4:
                vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", "seg_liver"))[0]
    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn


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


def FCN(inputs, number_slices=1, volume=False, scope='seg_liver'):
    """Defines the network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(scope, 'seg_liver', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
            net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # Get side outputs of the network
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None):
                side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')
                print (str(side_2)+'\n'+str(side_3)+'\n'+str(side_4)+'\n'+str(side_5)+'\n')
                # Supervise side outputs
                side_2_s = slim.conv2d(side_2, number_slices, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, number_slices, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, number_slices, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, number_slices, [1, 1], scope='score-dsn_5')
                print ('\n'+str(side_2_s)+'\n'+str(side_3_s)+'\n'+str(side_4_s)+'\n'+str(side_5_s)+'\n')

                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
                    side_2_s = slim.convolution2d_transpose(side_2_s, number_slices, 4, 2, scope='score-dsn_2-up')
                    print (side_2_s)
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_2-cr', side_2_s)
                    side_3_s = slim.convolution2d_transpose(side_3_s, number_slices, 8, 4, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_3-cr', side_3_s)
                    side_4_s = slim.convolution2d_transpose(side_4_s, number_slices, 16, 8, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_4-cr', side_4_s)
                    side_5_s = slim.convolution2d_transpose(side_5_s, number_slices, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_5-cr', side_5_s)

                    # Main output
                    side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi2-cr', side_2_f)
                    side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi3-cr', side_3_f)
                    side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi4-cr', side_4_f)
                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi5-cr', side_5_f)


                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], 3)

                net = slim.conv2d(concat_side, number_slices, [1, 1], scope='upscore-fuse')

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return net, end_points


number_slices = 3
output, _end_points = FCN(input_image, number_slices=number_slices)

initial_ckpt = os.path.join("../", 'train_files', 'vgg_16.ckpt')
init_weights = load_vgg_imagenet(initial_ckpt, number_slices)
Tensor_file = open("Tensor_Net", 'wb')
with tf.Session(config=config) as sess:
    init_weights(sess)
    sess.run(tf.global_variables_initializer())
    slim_FCN = sess.run([output,_end_points])
    print ("Result 4-D tensor", slim_FCN[0].shape)
    #print ("End_point = ", slim_FCN[1])
    result = slim_FCN[0]
    pickle.dump(result, Tensor_file)
    depth = result.shape[3]
    cv2.imshow("res_pic", matrix_res)
    result = result.astype(np.uint8)
    #RGB picture
    cv2.imshow("color pic", result[0, :, :, :])
    for i in range(depth):
        cv2.imshow("max_pool2d", result[0, :, :, i])
        cv2.waitKey(0)
    cv2.waitKey(0)
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())


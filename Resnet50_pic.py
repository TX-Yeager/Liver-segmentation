import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import scipy.io
import scipy.misc

import cv2
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
#matrix_res = cv2.imread("im1.png")
#matrix_res = cv2.imread("467.png")
#matrix_res = cv2.imread("467.png")
#matrix_res = cv2.imread("./tensorflow-fcn-master/test_data/tabby_cat.png")
#matrix = np.expand_dims(matrix_res, axis=0)
#matrix = matrix.astype(np.float32)
# for i in range(4):
#     matrix = np.append(matrix,matrix,axis=0)
# print (matrix.shape)
matrix = []
matrix_res = []
index = 437
for i in range(64):
    if i == 0:
        matrix = cv2.imread("../Mat_pic/17/horizontal_plane/" + str(index) + ".png")
        matrix = np.expand_dims(matrix, axis=0)
        matrix = matrix.astype(np.float32)
    else :
        matrix_res = cv2.imread("../Mat_pic/17/horizontal_plane/"+str(index)+".png")
        matrix_res = np.expand_dims(matrix_res, axis=0)
        matrix_res = matrix_res.astype(np.float32)
        matrix = np.append(matrix, matrix_res, axis=0)
    index += 1

input_image = tf.Variable(matrix)



tf.global_variables_initializer()
def det_lesion_arg_scope(weight_decay=0.0002):
    """Defines the arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer,
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc


def det_lesion_resnet(inputs, is_training_option=False, scope='det_lesion'):
    """Defines the network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """

    with tf.variable_scope(scope, 'det_lesion', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):

            net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=is_training_option)
            net = slim.flatten(net, scope='flatten5')
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.relu,
                                       weights_initializer=initializers.xavier_initializer(), scope='output')
            utils.collect_named_outputs(end_points_collection, 'det_lesion/output', net)

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return net, end_points


def load_resnet_imagenet(ckpt_path):
    """Initialize the network parameters from the Resnet-50 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()

    for v in var_to_shape_map:
        if "bottleneck_v1" in v or "conv1" in v:
            vars_corresp[v] = slim.get_model_variables(v.replace("resnet_v1_50", "det_lesion/resnet_v1_50"))[0]
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
    return init_fn


is_training = tf.placeholder(tf.bool, shape=())
number_slices = 3

output, _end_points = det_lesion_resnet(input_image, is_training_option=is_training)
init_weights = load_resnet_imagenet("../train_files/resnet_v1_50.ckpt")

#initial_ckpt = os.path.join("../", 'train_files', 'vgg_16.ckpt')
#init_weights = load_vgg_imagenet(initial_ckpt, number_slices)
Tensor_file = open("Tensor_Net", 'wb')
with tf.Session(config=config) as sess:
    init_weights(sess)
    sess.run(tf.global_variables_initializer())
    sess_resnet50 = sess.run([output,_end_points], feed_dict={is_training: False})
    print ("Result 4-D tensor", sess_resnet50[0].shape)
    #print (sess_resnet50)
    print (sess_resnet50[0])
    #print (sess_resnet50[1]("det_lesion\output"))
    #print (sess_resnet50[1]('det_lesion/out'))
    # print ("End_point = ", resnet[1])
    # result = resnet[0]
    # pickle.dump(result, Tensor_file)
    # #depth = result.shape[3]
    # cv2.imshow("res_pic", matrix_res)
    # result = result.astype(np.uint8)
    # #RGB picture
    # cv2.imshow("color pic", result[0, :, :, :])
    # for i in range(3):
    #     cv2.imshow("max_pool2d", result[0, :, :, i])
    #     cv2.waitKey(0)
    # cv2.waitKey(0)
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())


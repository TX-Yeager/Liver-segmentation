import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
from tensorflow.contrib.layers.python.layers import utils
import scipy
from dataset.dataset_seg import Dataset
import cv2
from scipy.io import loadmat
checkpoint_path ="/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/train_files/seg_liver_ck/networks/seg_liver.ckpt"
volume = False

result_path = "./result"

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

def seg_liver(inputs, number_slices=1, volume=False, scope='seg_liver'):
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

                # Supervise side outputs
                side_2_s = slim.conv2d(side_2, number_slices, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, number_slices, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, number_slices, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, number_slices, [1, 1], scope='score-dsn_5')
                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
                    side_2_s = slim.convolution2d_transpose(side_2_s, number_slices, 4, 2, scope='score-dsn_2-up')
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

def seg_liver_arg_scope(weight_decay=0.0002):
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

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                print ('input + output channels need to be the same')
                raise
            if h != w:
                print ('filters need to be square')
                raise
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors

def preprocess_img(image, number_slices):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
	Image ready to input the network (1,W,H,3)
    """
    images = [[] for i in range(np.array(image).shape[0])]

    if number_slices > 2:
        for j in range(np.array(image).shape[0]):
            if type(image) is not np.ndarray:
                for i in range(number_slices):
                    images[j].append(np.array(scipy.io.loadmat(image[0][i])['section'], dtype=np.float32))
            else:
                img = image
    else:
        for j in range(np.array(image).shape[0]):
            for i in range(3):
                images[j].append(np.array(scipy.io.loadmat(image[0][0])['section'], dtype=np.float32))
    in_ = np.array(images[0])
    in_ = in_.transpose((1, 2, 0))
    in_ = np.expand_dims(in_, axis=0)

    return in_

number_slices = 3

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.logging.set_verbosity(tf.logging.INFO)
# Input data
batch_size = 1
number_of_slices = number_slices
depth_input = number_of_slices
if number_of_slices < 3:
    depth_input = 3

input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

# Create the cnn
with slim.arg_scope(seg_liver_arg_scope()):
    net, end_points = seg_liver(input_image, number_slices, volume)
probabilities = tf.nn.sigmoid(net)
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create a saver to load the network
saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

test_file = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/seg_DatasetList/testing_volume_3 _3pic.txt"
database_root = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database"
dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(interp_surgery(tf.global_variables()))
    saver.restore(sess, checkpoint_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for frame in range(0, dataset.get_test_size()):
        img, curr_img = dataset.next_batch(batch_size, 'test')
        curr_ct_scan = curr_img[0][0].split('/')[-2]
        curr_frames = []
        if 1:
            for i in range(number_of_slices):
                curr_frames.append([curr_img[0][i].split('/')[-1].split('.')[0] + '.png'])
            if not os.path.exists(os.path.join(result_path, curr_ct_scan)):
                os.makedirs(os.path.join(result_path, curr_ct_scan))
            image = preprocess_img(curr_img, number_slices)



            #load .mat file
            path = "../LiTS_database/images_volumes/" + str(115) + "/"
            #mat_txt = open("./Mat2Txt/Matrix_" + str(115) + ".txt", mode="wb")
            m = loadmat("../LiTS_database/images_volumes/115/1.mat")
            matrix_res = np.array(m['section'], dtype=np.float32)
            # matrix_res = cv2.imread("./tensorflow-fcn-master/test_data/tabby_cat.png")
            matrix = np.expand_dims(matrix_res, axis=0)
            matrix = np.expand_dims(matrix, axis=3)
            image = np.append(matrix, matrix, axis=3)
            image = np.append(image, matrix,axis=3)

            res = sess.run(probabilities, feed_dict={input_image: image})
            res_np = res.astype(np.float32)[0, :, :, number_of_slices / 2]

            aux_var = curr_frames[number_of_slices / 2][0]
            scipy.misc.imsave(os.path.join(result_path, curr_ct_scan, aux_var), res_np)
            res_np = res.astype(np.float32)[0, :, :, 1]
            cv2.imwrite("./result/115/cv2_Save_"+str(aux_var),res_np*255)
            cv2.imshow("Save CV Processed CT",res_np)
            cv2.waitKey(0)

            # print ('Saving ' + os.path.join(result_path, curr_ct_scan, aux_var))
            #
            # for i in range(number_of_slices):
            #     aux_var = curr_frames[i][0]
            #     if not os.path.exists(os.path.join(result_path, curr_ct_scan, aux_var)):
            #         res_np = res.astype(np.float32)[0, :, :, i]
            #         scipy.misc.imsave(os.path.join(result_path, curr_ct_scan, aux_var), res_np)
            #         print ('Saving ' + os.path.join(result_path, curr_ct_scan, aux_var))
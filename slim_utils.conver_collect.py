import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.utils as utils
import numpy as np
# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(gpu_options=gpu_options)
from tensorflow.contrib.layers.python.layers import utils
import os
utils.collect_named_outputs()
utils.convert_collection_to_dict()
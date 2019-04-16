import tensorflow as tf
import numpy as np
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement = True

np.set_printoptions(threshold=np.inf)

matrix = cv2.imread("345.png",cv2.IMREAD_GRAYSCALE)
matrix2 = cv2.imread("194.png",cv2.IMREAD_GRAYSCALE)
x = tf.placeholder(tf.float32, shape=(512, 512))
y = tf.placeholder(tf.float32, shape=(512, 512))
#z = tf.matmul(y, x)
z = tf.add(y, x)
print (matrix)
#Set the gpu config
with tf.Session(config=config) as sess:
    #print(sess.run(y))
    #yarray = np.random.rand(512,512)
    yarray = np.eye(512)*25

    result = sess.run(z, feed_dict={x: matrix, y: matrix2})

    #change the np's type must use
    #Example:
    #matrix.astype(np.uint8)
    #matrix.dtype = 'uint8'dosen't work
    result = result.astype(np.uint8)
    #print (result)
    cv2.imshow("z", result)
    cv2.waitKey(0)
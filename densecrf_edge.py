import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from cv2 import imread, imwrite
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def image_hist(image):
    color = ('b', 'g', 'r')   #
    for i , color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])  #
        plt.plot(hist, color)
        plt.xlim([0, 256])
    plt.show()

def optimizer_CT_Blur(path):
    img = cv2.imread(path)

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    img = cv2.blur(gradient, (3, 3))

    # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    ret, res4_boneReduce_threshold = cv2.threshold(res2, 150, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow("x", res2)
    cv2.waitKey(0)

    # copy the data
    res4_temp = np.zeros(res4_boneReduce_threshold.shape, np.uint8)
    res4_temp = res4_boneReduce_threshold.copy()
    gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
    # copy the gray image because function
    gray_temp = gray.copy()
    # findContours will change the imput image into another

    ret, contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # #show the contours of the imput image
    # cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)
    #
    # #find the max area of all the contours and fill it with 0
    # area = []
    #
    # for i in range(len(contours)):
    #     area.append(cv2.contourArea(contours[i]))
    #
    # max_idx = np.argmax(area)
    # cv2.fillPoly(gray, contours[max_idx], 0)
    # cv2.imshow("fill",gray)
    # cv2.waitKey(0)

    c_max = []
    max_area = 0
    max_cnt = 0
    for x in range(len(contours)):
        cnt = contours[x]
        area = cv2.contourArea(cnt)
        # find max countour
        if (area >= max_area):
            if (max_area != 0):
                c_min = []
                c_min.append(max_cnt)
                cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)
            max_area = area
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            cv2.drawContours(res, c_min, -1, (0, 0, 0), cv2.FILLED)

    c_max.append(max_cnt)

    imag = cv2.drawContours(gray_temp, c_max, -1, (255, 255, 255), cv2.FILLED)

    th, im_th = cv2.threshold(imag, 200, 255, cv2.THRESH_BINARY)
    output_mask = im_th
    return output_mask


def optimizer_CT(path):
    img = cv2.imread(path)


    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # ret, res3_bone = cv2.threshold(img,100,225,cv2.THRESH_BINARY)

    # res4_boneReduce = (res3_bone-res2)

    # select the specific region you want to mask
    ret, res4_boneReduce_threshold = cv2.threshold(res2, 150, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow("x", res2)
    cv2.waitKey(0)

    # copy the data
    res4_temp = np.zeros(res4_boneReduce_threshold.shape, np.uint8)
    res4_temp = res4_boneReduce_threshold.copy()
    gray = cv2.cvtColor(res4_temp, cv2.COLOR_BGR2GRAY)
    # copy the gray image because function
    gray_temp = gray.copy()
    # findContours will change the imput image into another

    ret, contours, ret_hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # show the contours of the imput image
    cv2.drawContours(res4_temp, contours, -1, (0, 255, 255), 1)

    # find the max area of all the contours and fill it with 0
    area = []

    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(area)
    cv2.fillPoly(gray, contours[max_idx], 0)

    # cv2.fillConvexPoly(gray,contours[max_idx,0])
    # test = cv2.fillConvexPoly(gray, contours[max_idx], 0)
    # cv2.imshow("1", test)
    res4_boneReduce_threshold = cv2.cvtColor(res4_boneReduce_threshold, cv2.COLOR_BGR2GRAY)
    out_put = (res4_boneReduce_threshold - gray)

    # resr_boneReduce_threshold (512 512 3)
    # gray (512 512)

    th, im_th = cv2.threshold(out_put, 200, 255, cv2.THRESH_BINARY_INV);

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(out_put, mask, (0, 0), 255)
    out_put_liverMask = cv2.bitwise_not(out_put)
    return out_put_liverMask
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



# fn_im = "./406_res.png"#sys.argv[1]
# fn_anno = "./406Process.png"#sys.argv[2]
# fn_output = "x.png"#sys.argv[3]

fn_im = "./568_res.png"#sys.argv[1]
fn_anno = "./568mask.png"#sys.argv[2]
fn_output = "x.png"#sys.argv[3]



img = imread(fn_im)
img = rotate(img,270)
cv2.imshow("img",img)

# Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
anno_rgb = imread(fn_anno)
cv2.imshow("Annotation",anno_rgb)
#anno_rgb = cv2.blur(anno_rgb & img,(11,11))
#anno_rgb = anno_rgb & img
#anno_rgb = img & weights
cv2.imshow("anno_rgb",anno_rgb)
cv2.waitKey(0)
anno_rgb = anno_rgb.astype(np.uint32)
anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

# Convert the 32bit integer color to 1, 2, ... labels.
# Note that all-black, i.e. the value 0 for background will stay 0.
colors, labels = np.unique(anno_lbl, return_inverse=True)

# But remove the all-0 black, that won't exist in the MAP!
HAS_UNK = 10 in colors
if HAS_UNK:
    print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
    print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
    colors = colors[1:]
#else:
#    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

# And create a mapping back from the labels to 32bit integer colors.
colorize = np.empty((len(colors), 3), np.uint8)
colorize[:,0] = (colors & 0x0000FF)
colorize[:,1] = (colors & 0x00FF00) >> 8
colorize[:,2] = (colors & 0xFF0000) >> 16

# Compute the number of classes in the label image.
# We subtract one because the number shouldn't include the value 0 which stands
# for "unknown" or "unsure".
n_labels = len(set(labels.flat)) - int(HAS_UNK)
print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

###########################
### Setup the CRF model ###
###########################
use_2d = False
# use_2d = True
if use_2d:
    print("Using 2D specialized functions")

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
else:
    print("Using generic 2D functions")

    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


####################################
### Do inference and compute MAP ###
####################################

# Run five inference steps.
Q = d.inference(10)

# Find out the most probable class for each pixel.
MAP = np.argmax(Q, axis=0)

# Convert the MAP (labels) back to the corresponding colors and save the image.
# Note that there is no "unknown" here anymore, no matter what we had at first.
MAP = colorize[MAP, :]
output = MAP.reshape(img.shape)

#imwrite(fn_output, output)
cv2.imshow("output", output)
cv2.waitKey(0)


res = output

imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 150, 255, 0)
thresh = imgray
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#V3
c_max = []
max_area = 0
max_cnt = 0
for x in range(len(contours)):
    cnt = contours[x]
    area = cv2.contourArea(cnt)
    # find max countour
    if (area > max_area):
        if (max_area != 0):
            c_min = []
            c_min.append(max_cnt)
            cv2.drawContours(res, c_min, -1, (255, 255, 255), cv2.FILLED)
        max_area = area
        max_cnt = cnt
    else:
        c_min = []
        c_min.append(cnt)
        cv2.drawContours(res, c_min, -1, (255, 255, 255), cv2.FILLED)

c_max.append(max_cnt)
# cv2.fillPoly(res, c_max,[255,255,255])
# cv2.imshow("fillPoly",res)
# cv2.waitKey(0)

imag_green = cv2.drawContours(res, c_max, -1, (255, 255, 255), -1)
imag_green = cv2.blur(imag_green, (5, 5))

ret, imag_green = cv2.threshold(imag_green,128,255,cv2.THRESH_BINARY)

cv2.imshow("x", imag_green)
cv2.waitKey(0)
cv2.imshow("x",imag_green & img)
cv2.waitKey(0)
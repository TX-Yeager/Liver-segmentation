import numpy as np
import cv2
import os
import pydensecrf.densecrf as dcrf
from cv2 import imread, imwrite
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def Kmean(img):

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # classify color
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res_center = center[label.flatten()]
    res2_output = res_center.reshape((img.shape))

    # ret, res3_bone = cv2.threshold(img,100,225,cv2.THRESH_BINARY)

    # res4_boneReduce = (res3_bone-res2)

    # select the specific region you want to mask
    # ret, res4_boneReduce_threshold = cv2.threshold(res2,150,0,cv2.THRESH_TOZERO_INV)
    # cv2.imshow("res2", res2_output)
    # cv2.waitKey(0)
    return res2_output


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def densecrf_mask(img, anno_rgb):
    ##################################
    ### Read images and annotation ###
    ##################################
    #anno_rgb = imread(fn_anno).astype(np.uint32)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 10 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

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
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    output = MAP.reshape(img.shape)
    return output

patient_num = 117

def file_name(file_dir):
    L = []
    numofFile = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                numofFile += 1
                L.append(os.path.join(root, file))
                #print (file)
    return L,numofFile


_, numberofFile = file_name("../result/"+str(patient_num)+"Process/")
print (numberofFile)

path_ProcessMask = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/result/"+str(patient_num)+"ProcessMask/"
path_TruethMask = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/LiTS_database/liver_seg/"+str(patient_num)+"/"
path_TrueResult = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/result/"+str(patient_num)+"TrueResult/"
path_MaskMap = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/result/"+str(patient_num)+"Map/"
path_MaskMap_Result = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/result/"+str(patient_num)+"Densecrf_Map_Result/"
path_densecrfMask = "/home/chacky/PycharmProjects/untitled3/liverseg-2017-nipsws-master/tf_study/result/" + str(patient_num) + "DensecrfMask/"
if not os.path.exists(path_ProcessMask):
    os.mkdir(path_ProcessMask)
if not os.path.exists(path_MaskMap):
    os.mkdir(path_MaskMap)
if not os.path.exists(path_MaskMap_Result):
    os.mkdir(path_MaskMap_Result)
if not os.path.exists(path_TrueResult):
    os.mkdir(path_TrueResult)
if not os.path.exists(path_densecrfMask):
    os.mkdir(path_densecrfMask)
for i in range(1, numberofFile):
    #i=568
    print("CT number "+str(i))
    res = cv2.imread("../../Mat_pic/"+str(patient_num)+"/horizontal_plane/"+str(i)+".png")
    res_copy = cv2.imread("../result/"+str(patient_num)+"Process/"+str(i)+".png")

    res_true = cv2.imread(path_TruethMask + str(i) + ".png")
    #mask_tf = cv2.imread("../result/"+str(patient_num)+"Process/"+str(i)+".png")
    mask_tf = cv2.imread("../result/" + str(patient_num) + "/" + str(i) + ".png")
    mask_tf = rotate(mask_tf, 90)
    res_true = rotate(res_true, 90)
    #mask = rotate(mask,90)
    Image = mask_tf & res
    cv2.imwrite(path_MaskMap+str(i)+".png", Image)

    Image_True = res_true & res
    cv2.imwrite(path_TrueResult + str(i) + ".png", Image_True)
    try:
        mask_tf = mask_tf.astype(np.uint32)
        result = densecrf_mask(res, mask_tf)
        result = rotate(result, 270)
        ret, result = cv2.threshold(result, 128, 256, cv2.THRESH_BINARY)

    except:
        th, result = cv2.threshold(res_copy, 100, 255, cv2.THRESH_BINARY)
        result = rotate(res_copy, 270)
        cv2.imwrite("../result/" + str(patient_num) + "DensecrfMask/" + str(i) + ".png", res_copy)
        res = cv2.imread("../../Mat_pic/" + str(patient_num) + "/horizontal_plane/" + str(i) + ".png")
        mask_tf = result
        mask_tf = rotate(mask_tf, 90)
        # mask = rotate(mask,90)
        Image = mask_tf & res
        cv2.imwrite(path_MaskMap_Result + str(i) + ".png", Image)
        continue

    cv2.imwrite("../result/"+str(patient_num)+"DensecrfMask/"+str(i)+".png",result)

    res = cv2.imread("../../Mat_pic/" + str(patient_num) + "/horizontal_plane/" + str(i) + ".png")
    mask_tf = result
    mask_tf = rotate(mask_tf, 90)
    # mask = rotate(mask,90)
    Image = mask_tf & res
    cv2.imwrite(path_MaskMap_Result + str(i)+".png", Image)
    #cv2.imshow("result :"+str(i),result)
    #cv2.waitKey(0)
    #break

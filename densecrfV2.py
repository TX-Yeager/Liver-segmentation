import cv2

res = cv2.imread("568.png")
res_mask = cv2.imread("568mask.png")


cv2.imshow("+",res + res_mask)
cv2.imshow("-",res - res_mask)
cv2.imshow("- n",res_mask - res)
cv2.imshow("&",res & res_mask)
cv2.imshow("|",res | res_mask)
cv2.imwrite("568plusmask.png",res+res_mask)
cv2.waitKey(0)
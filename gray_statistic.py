import cv2
import numpy as np
import matplotlib.pyplot as plt

ax=plt.subplot(1,1,1)

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def image_hist(ax,image,color_show):
    color = ('b', 'g', 'r')   #
    for i , color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [5, 256])  #
        ax.plot(hist, color,color = color_show,lw = 1)
        plt.xlim([50, 256])
    #plt.show()

img = cv2.imread('./568TrueMap.png')
#img2 = cv2.imread('./568ProceMap.png')
img2 = cv2.imread('./map result.png')
#plot_demo(img)
image_hist(ax,img , 'r')
image_hist(ax,img2, 'b')

p2=ax.plot([3,2,1], label="Predict")
p1=ax.plot([1,2,3], label="Groud Truth")
ax.set_xlabel('gray level')
ax.set_title('histogram')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
plt.show()



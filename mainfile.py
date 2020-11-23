import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as transform
import bound as bo

def main():
    foreground=bo.get_foreground()
    masknew=foreground.copy()
    masknew[masknew>0]=1

    background=io.imread('/home/jayasurya/Desktop/storerack.jpg')/255.0
    background = transform.resize(background, foreground.shape[:2])
    background = background*(1 - masknew)

    composed_image = background + foreground

    plt.imshow(composed_image)
    plt.axis('off')
    plt.show()
main()

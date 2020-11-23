import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from skimage import io

def extract(image, coordinates):
	img = io.imread(image)
	mask = np.zeros(img.shape[:2], np.uint8)
	bgModel = np.zeros((1,  65), np.float64)
	fgModel = np.zeros((1,  65), np.float64)
	rect = tuple(coordinates)
	cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	img_1 = img*mask2[:, :, np.newaxis]
	return img_1
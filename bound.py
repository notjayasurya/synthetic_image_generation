from imageai.Detection import ObjectDetection
import os
import cv2
import foreground_extration as fe
import tensorflow as tf
import skimage.io as io

INPUT_IMAGE = "cornflakes.jpg"
OUTPUT_IMAGE = "edited_cornflakes.jpg"

def get_rect():
	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath(os.path.join("/home/jayasurya/Downloads", 'resnet50_coco_best_v2.0.1.h5'))
	detector.loadModel()
	detections = detector.detectObjectsFromImage(input_image=os.path.join("/home/jayasurya/Desktop/", INPUT_IMAGE), output_image_path= os.path.join("/home/jayasurya/Desktop/", OUTPUT_IMAGE))
	return detections

def get_foreground():
	detections = get_rect()
	# print(f"Name = {detections[0]['name']}\nCo-ordinates = {detections[0]['box_points']}")
	image = fe.extract("/home/jayasurya/Desktop/cornflakes.jpg",detections[0]['box_points'])/255.0
	io.imsave("/home/jayasurya/Desktop/cropped_image/cropped_img_cornflakes.jpg",image)
	return image

#cv2.imshow("Extracted Image", get_foreground())
#cv2.imwrite('car_image', get_foreground())
#cv2.waitKey(0)

import cv2
img = cv2.imread("/home/jayasurya/Desktop/tesla_img.jpg")
img_size = cv2.resize(img,(256,256))
cv2.imwrite("/home/jayasurya/Desktop/tesla_img_rewrite.jpg",img_size)

import cv2
import os


current_dir = os.path.dirname(__file__)
image_dir = current_dir + '/new_images'

image_path = image_dir + '/1-lighted_new_images_1.jpg'

image = cv2.imread(image_path)
cv2.imshow("original_image", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
out_gray=cv2.divide(image, bg, scale=255)
out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

cv2.imshow('binary', out_binary)  


cv2.imshow('gray', out_gray)  



"""
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaussian_threshold = cv2.adaptiveThreshold(gray_image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("treshold_img", gaussian_threshold)

mean_threshold = cv2.adaptiveThreshold(gray_image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

ret3,th3 = cv2.threshold(mean_threshold,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("OTSU_image", th3)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
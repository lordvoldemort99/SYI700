import cv2
import os
import matplotlib.pyplot as plt

image_path = "/1.jpg"

img = cv2.imread(image_path)

#### image processing and edge detection
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow("image", img)


# cv2.imshow("blured image", img)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
# cv2.imwrite("adaptive_threshold.jpg", img)

sobel_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=13)
# cv2.imwrite("sobel.jpg", sobel_x)

canny = cv2.Canny(image=img, threshold1=100, threshold2=200)
# cv2.imwrite("canny.jpg", canny)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure()
### B_bottle
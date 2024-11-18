import numpy as np
import cv2

def nothing(x):
    pass

cv2.namedWindow("TrackBar")#<<< needs to be completet*******************************************************
cv2.createTrackbar("Threshold1", "TrackBar", 0, 255, nothing)#<<< needs to be completet*********************
cv2.createTrackbar("Threshold2", "TrackBar", 0, 255, nothing)#<<< needs to be completet*********************
cv2.createTrackbar("GaussianBlur1", "TrackBar", 0, 255, nothing)#<<< needs to be completet******************
cv2.createTrackbar("GaussianBlur2", "TrackBar", 0, 255, nothing)#<<< needs to be completet******************
cv2.createTrackbar("Iterations", "TrackBar",0,255,nothing)

image_path = "./images/bottle_image4.jpg"
    
while True:
    img=cv2.imread(image_path)
    img=cv2.resize(img, (960, 540))

    #img_copy=img.copy()
    ## lower threshold for Canny
    thres1=cv2.getTrackbarPos("Threshold1", "TrackBar")#<<< needs to be completet**************************
    ## Upper threshold for Canny
    thres2=cv2.getTrackbarPos("Threshold2", "TrackBar")#<<< needs to be completet**************************
    #note that gausian blur has a special condition to work with trackbars! 
    GaussianBlur1=cv2.getTrackbarPos("GaussianBlur1", "TrackBar")#<<< needs to be completet ***************
    GaussianBlur2=cv2.getTrackbarPos("GaussianBlur2", "TrackBar")#<<< needs to be completet ***************

    ## For Dilate
    SetIterations=cv2.getTrackbarPos("Iterations", "TrackBar")#<<< needs to be completet ******************

    gray_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#<<< needs to be completet ****************************

    ## To achieve a proper Gaussian blur, the kernel size needs to be an odd number.
    ## This is because the kernel is centered around the current pixel, and using 
    ## an odd-sized kernel ensures that the current pixel is at the center of the kernel.
    image_kernel = (GaussianBlur1 * 2 + 1,GaussianBlur2 * 2 + 1)
    blurred=cv2.GaussianBlur(gray_image, image_kernel, 0)#<<< needs to be completet***********************
    print("blurred",blurred.shape)

    ## Finding Edges
    edged=cv2.Canny(blurred, thres1, thres2)#<<< needs to be completet ***********************************

    ## Thicken the edges
    dilalted=cv2.dilate(edged, image_kernel, iterations=SetIterations)#<<< needs to be completet *********
    
    ## draw a rectangle around edges
    countours,heirarchy = cv2.findContours(dilalted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#<<< needs to be completet
    color_image = cv2.cvtColor(dilalted, cv2.COLOR_GRAY2RGB)
    color_image = cv2.drawContours(color_image, countours, -1, (0, 255, 0), 3)
    # coun = cv2.drawContours(dilalted, countours, -1, (0, 255, 0), 3) 
    cv2.imshow("image", color_image)
    # cv2.waitKey(1)


    #Convert single channel pictures to three channel pictures to be able to handle any new input from RGB. 
    # This is the same channel 3 times in the matrix.
    # array([[1, 4],
    #   [2, 5],
    #   [3, 6]])
    # 
    # for more info: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    

    gray = np.stack(gray_image, axis=-1) #<<< needs to be completet
   
    blurred = np.stack(blurred, axis=-1)    #<<< needs to be completet
    edged = np.stack(edged, axis=-1)#<<< needs to be completet
    dilalted = np.stack(dilalted, axis=-1)#<<< needs to be completet

    color_image = np.stack(color_image, axis=-1)
    #we then put the images into  a list to be displayed beside eacother
    images=[gray,blurred,edged,dilalted]#<<< needs to be completet
    win_names = ["gray","blurred","edged","dilalted"]

    img_stack=np.hstack(images)#<<< needs to be completet
    img_stack=cv2.resize(img_stack, (960,540 ))#<<< needs to be completet
    

    
    cv2.imshow("bilder",img_stack)  
    # cv2.imshow("Gray",gray)
    # cv2.imshow("Blurred",blurred)
    # cv2.imshow("Edged",edged)
    # cv2.imshow("Dialted",dilalted)

    print("objects found: ", len(countours))

    cv2.waitKey(300)


        ######Answer#####
# what are the good trackbar values for finding the objects in the image? 
## For Canny filter:
# Thresh1 = 57
# Thresh2 = 57
# gaussian1 = 9
# gaussian2 = 5
# iteration = 12
#################
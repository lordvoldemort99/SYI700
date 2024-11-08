from pypylon import pylon
import cv2
import numpy as np
import os
import time
import sys

sys.path.append("./")
from keypoint_detection.yolo.bottle_finder import keypoint

keypoint_detector = keypoint()

current_dir = os.path.dirname(__file__)
predictedImg_savePath = current_dir + '/predicted_images/' 

# Create an instant camera object with the first camera device found
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Start grabbing continuously (camera will start to capture images)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Convert images to OpenCV format and display
converter = pylon.ImageFormatConverter()
# Convert to OpenCV BGR format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
image_counter = 0
last_time = 0
while camera.IsGrabbing():
   
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    new_time = time.time() 
    if grabResult.GrabSucceeded():
        # Access the image data as a NumPy array
        image = converter.Convert(grabResult)
        img = image.GetArray()
        image_h, image_w, _ = img.shape
        img = cv2.resize(img, (image_w//4, image_h//4))

        key_point_data = keypoint_detector.predict(img)
        print(f"key points are: {key_point_data}")
        keypoint_detector.show_image_with_keypoints()


        key = cv2.waitKey(1) & 0xFF
        # Break loop if 'ESC' is pressed
        if key == 27:
            break
        elif key == ord('s'):
            image_counter += 1
            keypoint_detector.save_predicted_image(predictedImg_savePath, f'prediction_{image_counter}.jpg')
            
            print(f"********** Image {image_counter} Saved **********")
    
    grabResult.Release()

camera.StopGrabbing()
cv2.destroyAllWindows()

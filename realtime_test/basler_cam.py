from pypylon import pylon
import cv2
import numpy as np
import os
import time

current_dir = os.path.dirname(__file__)

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
        # Display the image in a window
        cv2.imshow('resized Camera', img)

        if new_time - last_time > 0.5:
            ## take pictire
            image_counter += 1
            cv2.imwrite(f"{current_dir}/{image_counter}_images.jpg", img)
            print("********** Image taken **********")
            last_time = new_time

        if cv2.waitKey(1) & 0xFF==ord('s'):
            image_counter += 1
            print(image_counter)
            cv2.imwrite(f"{current_dir}/{image_counter}_blue.jpg", img)
            print("********** Image Saved **********")
        # Break loop if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    grabResult.Release()

# Stop camera capturing
camera.StopGrabbing()

# Release camera and close windows
cv2.destroyAllWindows()

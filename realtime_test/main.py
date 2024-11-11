from pypylon import pylon
import cv2
import numpy as np
import os
import time
import sys

import coordinate_translator

sys.path.append("./")
from keypoint_detection.yolo.bottle_finder import keypoint

image_counter = 0
last_time = 0
record_video = False
FPS = 10
ARUCO_LENGTH = 100
ARUCO_MARKER = cv2.aruco.DICT_5X5_100

current_dir = os.path.dirname(__file__)
predictedImg_savePath = current_dir + '/predicted_images/' 

translator = coordinate_translator.translator(ARUCO_MARKER, ARUCO_LENGTH)

keypoint_detector = keypoint()

# Convert images to OpenCV format and display
converter = pylon.ImageFormatConverter()
# Convert to OpenCV BGR format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Create an instant camera object with the first camera device found
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Start grabbing continuously (camera will start to capture images)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

frame_width = (camera.Width.Value)//4
frame_height = (camera.Height.Value)//4

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(f'{current_dir}/videos/prediction_video.mp4', fourcc, FPS, (frame_width, frame_height))

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    new_time = time.time() 
    if grabResult.GrabSucceeded():
        # Access the image data as a NumPy array
        image = converter.Convert(grabResult)
        img = image.GetArray()
        img = cv2.resize(img, (frame_width, frame_height))

        bottle_data = keypoint_detector.bottle_features(image)
        pick_point = tuple(bottle_data[:2])
        bottle_orientation = bottle_data[2]
        print("bottle key-points: ", bottle_data)
        print(f"pick point: {pick_point}")

        new_coordinates, new_orientation = translator.translate_coordinate(image, pick_point, bottle_orientation)
        new_coordinates = [coords[0] for coords in new_coordinates.tolist()]
        new_coordinates = (new_coordinates[:2])
        print("new_coordinates: ", new_coordinates, "real_angle: ", new_orientation) ### new_coordinates in mm and angle in degrees

        keypoint_detector.show_image_with_keypoints()

        if record_video == True:
            print("recording ...........")
            out.write(keypoint_detector._image)

        key = cv2.waitKey(1) & 0xFF
        # Break loop if 'ESC' is pressed
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            image_counter += 1
            keypoint_detector.save_predicted_image(predictedImg_savePath, f'prediction_{image_counter}.jpg')
            
            print(f"********** Image {image_counter} Saved **********")
        elif key == ord('r'):
            print("record started")
            record_video = True
    
    grabResult.Release()

camera.StopGrabbing()
camera.Close()
out.release()
cv2.destroyAllWindows()



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

### these have been found by avaluating by measuring in the real layout
X_OFFSET = 20
Y_OFFSET = 20

current_dir = os.path.dirname(__file__)
savePath = current_dir + '/predicted_images' 

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
        image = image.GetArray()
        image = cv2.resize(image, (frame_width, frame_height))

        if new_time - last_time > 0.5:
            aruco_state = translator.aruco_detector(image)
            bottle_data = keypoint_detector.bottle_features(image)
            if bottle_data != None:
                pick_point = tuple(bottle_data[:2])
                bottle_orientation = bottle_data[2]
                print("bottle key-points: ", bottle_data)
                print(f"pick point: {pick_point}")

                if aruco_state != coordinate_translator.NO_ARUCO_FOUND:
                    new_coordinates, new_orientation = translator.translate_coordinates(pick_point, bottle_orientation)
                    new_coordinates = [coords[0] for coords in new_coordinates.tolist()]
                    new_coordinates = (new_coordinates[:2])
                    new_coordinates[0] = new_coordinates[0] - X_OFFSET
                    new_coordinates[1] = new_coordinates[1] - Y_OFFSET
                
                    print("new_coordinates: ", new_coordinates, "real_angle: ", new_orientation) ### new_coordinates in mm and angle in degrees

                    cv2.putText(image, (f"real_pick_coords: ({int(new_coordinates[0])}, {int(new_coordinates[1])})"), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            last_time = new_time
        # translator.show_aruco()
        keypoint_detector.show_image_with_keypoints()
        # cv2.imshow("image", image)
        if record_video == True:
            print("recording ...........")
            out.write(keypoint_detector._image)

        key = cv2.waitKey(1) & 0xFF
        # Break loop if 'ESC' is pressed
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            image_counter += 1
            cv2.imwrite(f'{savePath}/image_{image_counter}.jpg', keypoint_detector._image)
            
            print(f"********** Image {image_counter} Saved **********")
        elif key == ord('r'):
            print("record started")
            record_video = True
    
    grabResult.Release()

camera.StopGrabbing()
camera.Close()
out.release()
cv2.destroyAllWindows()



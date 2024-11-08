from pypylon import pylon
import cv2
import os
import time

counter = 0
constant_saving_counter = 0
last_time = 0
interval_for_saving_images = 0.5

current_dir = os.path.dirname(__file__)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

camera.Open()
camera.PixelFormat.SetValue("BGR8")  # Ensure the pixel format is compatible with OpenCV

# Set video properties
frame_width = camera.Width.Value
frame_height = camera.Height.Value
fps = 20  

# Initialize VideoWriter for saving video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(f'{current_dir}/videos/output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Start grabbing frames from the camera
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

try:
    print("Recording video. Press 'q' to stop.")
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        new_time = time.time() 
        if grab_result.GrabSucceeded():
            # Convert frame to numpy array (compatible with OpenCV)
            frame = grab_result.Array
            frame = cv2.resize(frame, (frame_width//4, frame_height//4))
            # Display the frame (optional)
            cv2.imshow('Recording', frame)

            # Save the frame to the video file
            out.write(frame)

            if new_time - last_time > interval_for_saving_images:
                ## take pictire
                constant_saving_counter += 1
                cv2.imwrite(f"{current_dir}/images/{constant_saving_counter}_Videoimages.jpg", frame)
                print("********** Image taken **********")
                last_time = new_time


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                counter += 1
                print(f"Saving frame {counter}")
                cv2.imwrite(f"{current_dir}/images/videoframe{counter}.jpg", frame)

        grab_result.Release()

except KeyboardInterrupt:
    print("Recording interrupted by user.")

# Stop grabbing, release resources, and close camera
camera.StopGrabbing()
camera.Close()
out.release()
cv2.destroyAllWindows()

print("Video saved as 'output_video.mp4'")

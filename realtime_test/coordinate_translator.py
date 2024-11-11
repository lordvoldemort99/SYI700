import cv2
import numpy as np
import os

NO_ARUCO_FOUND = -1

class translator:
    def __init__(self, aruco_dictionary, aruco_length):
        """
            Parameter:
                aruco_dictionary : ex: cv2.aruco.DICT_5X5_100
                aruco_length : in mm
        """
        self.aruco_area = 4 * aruco_length
        self.aruco_marker = aruco_dictionary
    
    def translate_coordinate(self, image, coordinates, angle):
        old_x, old_y = coordinates

        font_color = {'red':(0, 0, 255), 'green': (0, 255, 0),'orange':(0, 100, 200), 'black':(0,0,0)}

        parameters = cv2.aruco.DetectorParameters()
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_marker)
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = aruco_detector.detectMarkers(image)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            if corners:
                int_corners = np.int32(corners) 

                aruco_perimeter = cv2.arcLength(corners[0], True)
                pixel_cm_ratio = aruco_perimeter / self.aruco_area

                aruco_top_left_x = int_corners[0][0][0][0] ### The point marked by drawDetectedMarkers
                aruco_top_left_y = int_corners[0][0][0][1]

                aruco_top_right_x = int_corners[0][0][1][0] ### the point in front of the marked point from the drawDetectedMarkers
                aruco_top_right_y = int_corners[0][0][1][1]

                aruco_bottom_right_x = int_corners[0][0][2][0] 
                aruco_bottom_right_y = int_corners[0][0][2][1]

                aruco_bottom_left_x = int_corners[0][0][3][0]
                aruco_bottom_left_x = int_corners[0][0][3][1]

                cv2.circle(image, (aruco_top_right_x, aruco_top_right_y), 2, font_color['orange'], 2)
                cv2.putText(image, 'topRight', (aruco_top_right_x, aruco_top_right_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color['orange'], 2)

                cv2.circle(image, (aruco_top_left_x, aruco_top_left_y), 2, font_color['orange'], 2)
                cv2.putText(image, 'topLeft', (aruco_top_left_x, aruco_top_left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color['orange'], 2)

                cv2.circle(image, (aruco_bottom_right_x, aruco_bottom_right_y), 2, font_color['orange'], 2)
                cv2.putText(image, 'topLeft', (aruco_bottom_right_x, aruco_bottom_right_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color['orange'], 2)

                ### calculat the orientation of the camera
                aruco_vector = np.array([aruco_top_right_x - aruco_top_left_x, aruco_top_right_y - aruco_top_left_y]) ### x direction
                image_vector = np.array([1,0])  ### (x, y)

                print(f"top_left: {aruco_top_left_x}, {aruco_top_left_y}; top_right: {aruco_top_right_x}, {aruco_top_right_y}")
                tfangle = np.arccos((np.dot(aruco_vector, image_vector)/ (np.linalg.norm(aruco_vector)*np.linalg.norm(image_vector))))   ## cos(theta)=A.B / |A|x|B|
                
                if aruco_top_right_y < aruco_top_left_y:
                    tfangle = -tfangle
                ### tfangle is in radian
                print(f"the angle between aruco upper side and image top line is {np.degrees(tfangle)}")

                tfmatrix1 = np.matrix([[np.cos(tfangle), -np.sin(tfangle), 0 , aruco_top_left_x],
                                       [np.sin(tfangle),  np.cos(tfangle), 0 , aruco_top_left_y],
                                       [       0       ,         0       , 1 ,        0        ],
                                       [       0       ,         0       , 0 ,        1        ]])
                
                old_coordinate = np.matrix([[old_x],[old_y],[0],[1]])
                new_coordinate = np.matmul(np.linalg.inv(tfmatrix1),old_coordinate)
                new_coordinate = new_coordinate/pixel_cm_ratio
                new_angle = angle + np.degrees(tfangle)
                return new_coordinate, new_angle
        else:
            return NO_ARUCO_FOUND



if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    image_path = current_dir + '/7_ArUco.jpg'
    image = cv2.imread(image_path)
    old_coordinate = (20, 30)
    angle = 20

    aruco_length = 100
    translator = translator(aruco_dictionary=cv2.aruco.DICT_5X5_100, aruco_length=aruco_length)
    translator_result = translator.translate_coordinate(image, old_coordinate, angle)

    if translator_result is NO_ARUCO_FOUND:
        print("NO ARUCO FOUND !!")
    else:
        print(f"new coordinates: {translator_result}")

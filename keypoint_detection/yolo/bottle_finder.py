from ultralytics import YOLO
import cv2
import os

class keypoint:
    def __init__(self):
        _current_dir = os.path.dirname(__file__)
        _model_path = _current_dir + '/best.pt'
        self.model = YOLO(_model_path)

        self.image_path = ''

    def predict(self, image_array) -> dict:
        '''
            Prameters:
            image_array : array of image

            Returns: 
            dict: {'cap_corner':0, 'pick_point':0, 'angle_point':0}\n
                  cap_corner: coordinates of bottle's cap top-left corner (for color detection), \n
                  pick_point: coordinates of center point under the cap (pick point), \n
                  angle_point: coordinates of center point at the end of the bottle(for angle caculation)
        '''
        self._image = image_array
        _points = []
        self.bottle_keypoints = {'cap_corner':0, 'pick_point':0, 'angle_point':0}

        _results = self.model(self._image)[0]

        ### bottle found
        if len(_results) > 0:
            for result in _results:
                for key_points in (result.keypoints.xy.tolist()):
                    _points.append(key_points)
            self.bottle_keypoints['cap_corner'] = _points[0][0]
            self.bottle_keypoints['pick_point'] = _points[0][1]
            self.bottle_keypoints['angle_point'] = _points[0][2]

            return self.bottle_keypoints
        ### No bottle found
        else: 
            return None 

    def show_image_with_keypoints(self):
        # self._image = cv2.imread(self.image)
        if any(value != 0 for value in self.bottle_keypoints.values()):
            for keypoint in self.bottle_keypoints.items():
                ### put the key points on the image
                keypoint_id = keypoint[0]
                x, y = int(keypoint[1][0]), int(keypoint[1][1])
                cv2.putText(self._image, f"{keypoint_id}: x:{x}, y:{y}", (x, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(self._image, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("predicted image", self._image)
        # cv2.waitKey(0)

    def save_predicted_image(self, save_path, image_name):
        cv2.imwrite(f"{save_path}/{image_name}.jpg", self._image)


'''
************************
TO DO
1. test what is the output when there is no bottle in the image ----- DONE

3. test the whole class    ----- DONE

2. test for the camera images - real time
4. test the show funtion when using the camer - the waitkey part might be a problem

'''

if __name__ == "__main__":
    image_name = 'emptyscene.jpg'
    # image_name = '67_red.jpg_0_1958.jpg'

    current_dir = os.path.dirname(__file__)
    image_path = current_dir + '/test_images/' + image_name
    save_path = current_dir + '/predicted_images'

    image = cv2.imread(image_path)
    key_point_detector = keypoint()
    result = key_point_detector.predict(image)
    print(f"result is : {result}")

    key_point_detector.show_image_with_keypoints()

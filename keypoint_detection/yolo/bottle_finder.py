from ultralytics import YOLO
import cv2
import os

class keypoint:
    def __init__(self):
        _currect_dir = os.path.dirname(__file__)
        _model_path = _currect_dir + '/best.pt'
        self.model = YOLO(_model_path)

        self.image_path = ''

    def predict(self, image_path) -> dict:
        '''
            Prameters:
            image_path (str): absolute path to the image or camera output

            Returns: 
            dict: {'cap_corner':0, 'pick_point':0, 'angle_point':0}\n
                  cap_corner: coordinates of bottle's cap top-left corner (for color detection), \n
                  pick_point: coordinates of center point under the cap (pick point), \n
                  angle_point: coordinates of center point at the end of the bottle(for angle caculation)
        '''
        self.image_path = image_path
        _points = []
        self.bottle_keypoints = {'cap_corner':0, 'pick_point':0, 'angle_point':0}

        _results = self.model(self.image_path)[0]

        for result in _results:
            for key_points in (result.keypoints.xy.tolist()):
                _points.append(key_points)
        self.bottle_keypoints['cap_corner'] = _points[0][0]
        self.bottle_keypoints['pick_point'] = _points[0][1]
        self.bottle_keypoints['angle_point'] = _points[0][2]

        return self.bottle_keypoints

    def show_image_with_keypoints(self):
        for keypoint in self.bottle_keypoints.items:
            self._image = cv2.imread(self.image_path)

            keypoint_id = keypoint[0]
            x, y = int(keypoint[1][0]), int(keypoint[1][1])
            cv2.putText(self._image, f"{keypoint_id}: x:{x}, y:{y}", (x, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(self._image, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow("predicted image", self._image)
            cv2.waitKey(0)

    def save_predicted_image(self, save_path, image_name):
        cv2.imwrite(f"{save_path}/{image_name}.jpg", self._image)


'''

************************
TO DO
1. test what is the output when there is no bottle in the image
2. test for the camera images - real time
3. test the whole class
4. test the show funtion when using the camer - the waitkey part might be a problem


image_name = '67_red.jpg_0_484.jpg'
test_image_path = currect_dir + '/test_images/' + image_name
save_path = currect_dir + '/predicted_images'

image = cv2.imread(test_image_path)

model = YOLO(model_path)
results = model(test_image_path)[0]

points = []
bottle_keypoints = {'cap_corner':0, 'pick_point':0, 'angle_point':0}

for result in results:
    for key_points in (result.keypoints.xy.tolist()):
        points.append(key_points)
        for keypoint_indx, keypoint in enumerate(key_points):
            x = int(keypoint[0])
            y = int(keypoint[1])
            cv2.putText(image, f"{keypoint_indx}: x:{x}, y:{y}", (x, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
print(points)
bottle_keypoints['cap_corner'] = points[0][0]
bottle_keypoints['pick_point'] = points[0][1]
bottle_keypoints['angle_point'] = points[0][2]

print(f"bottle points: {bottle_keypoints}")

cv2.imwrite(f"{save_path}/predicted_{image_name}.jpg", image)
# cv2.imshow("image", image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''
dict_hi = {'cap_corner':10, 'pick_point':50, 'angle_point':30}
for key in dict_hi.items():
    print(key[1])
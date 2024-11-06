from ultralytics import YOLO
import cv2
import os

currect_dir = os.path.dirname(__file__)
model_path = currect_dir + '/best.pt'

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
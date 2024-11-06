## Content
+ [Project Description](#project-description)
+ [Code Help](#help-for-the-code)
    + [Annotation tools](#annotation-tool)
    + [Connecting Google Colab to Google Drive](#connecting-google-colab-to-google-drive)
    + [fine tune YOLO11n-pose](#fine-tune-yolo11n-pose)
    + [Save the best mdoel during the train](#saving-the-best-model-during-the-train)
    + [Save the train result on Google Drive](#save-the-result-on-google-drive)
    + [Test the mdoel on new images](#predict-new-images)


### Project Description




### Code Help

#### Annotation tool

[online - CVAT.ai](https://www.cvat.ai/)

> Save the annotated data in `.xml` file.
> + This data will be one line contaning all images data. 
>
> Then it is needed to be converted to `coco` format to be used for YOLO.
> [Convert xml to coco](/keypoint_detection/bottle_neck_keypoint/annotations/cvat_to_coco.py)
 
on-device ***labelme*** : 
```shell
$ pip install labelme
$ labelme
```

### connecting Google Colab to Google Drive

```shell
from google.colab import drive

drive.mount('/content/gdrive')
```

#### fine tune YOLO11n-pose

- only one class is accepted in YOLO-pose, so far.
- the label file should be `.txt` and contains one line : `0 bx by bw bh x y` where `0` is the classs ID `bx by` are center point coordinates of the bounding box around the object, `bw bh` are bounding box width and height, `x y` are the position of the key point. If you have more than one key point, all their `x y` should be appended to this line.

#### Saving the best model during the train

> you need to turn the flag `save` to true in this line:

```py
model.train(data='/content/gdrive/My Drive/university_west_bottle_pickPoint/config.yaml', epochs=100, imgsz=(640, 480), save=True)

```

### Save the result on Google Drive

```shell
!scp -r /content/runs '/content/gdrive/My Drive/university_west_bottle_pickPoint'

```

### predict new images

```py
from ultralytics import YOLO
import numpy as np
import os

currect_dir = os.path.dirname(__file__)
model_path = currect_dir + '/best.pt'
test_image_path = currect_dir + '/test_images/4_yellow.jpg_0_85.jpg'

model = YOLO(model_path)
results = model(test_image_path)[0]

for result in results:
    for key_point in result.keypoints:
        print(key_point)
```

> The result of this code is this:
```shell
image 1/1 e:\SHiTU\programming\university_west_programming\integerating_sysetems_Vision\keypoint_detection\yolo\test_images\4_yellow.jpg_0_85.jpg: 544x640 1 pick_point, 150.0ms
Speed: 5.0ms preprocess, 150.0ms inference, 2.0ms postprocess per image at shape (1, 3, 544, 640)
ultralytics.engine.results.Keypoints object with attributes:      

conf: None
data: tensor([[[449.8272, 169.6225],
         [441.8085, 191.5298],
         [339.0192, 285.0518]]])
has_visible: False
orig_shape: (512, 612)
shape: torch.Size([1, 3, 2])
xy: tensor([[[449.8272, 169.6225],
         [441.8085, 191.5298],
         [339.0192, 285.0518]]])
xyn: tensor([[[0.7350, 0.3313],
         [0.7219, 0.3741],
         [0.5540, 0.5567]]])
```

> For getting only xy coordinates, make this change in the for loop:

```py
result.keypoints.xy.tolist()
```
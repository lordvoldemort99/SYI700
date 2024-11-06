import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

images_path = os.path.dirname(__file__) + '/images'
image_save_path = os.path.dirname(__file__) + '/augmented_images'

datagen = ImageDataGenerator(
                brightness_range=(0.5, 1.5),
                rotation_range=50,
                width_shift_range=0.05,
                height_shift_range=0.05,
                # rescale=1./255,
                shear_range=0.2,
                # zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')


for image_name in os.listdir(images_path):
    img_path = os.path.join(images_path, image_name)
    img = load_img(img_path)  # this is a PIL image
    image_array = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    image = image_array.reshape((1,) + image_array.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(image, batch_size=1,
                            save_to_dir=image_save_path, save_prefix=image_name, save_format='jpg'):
        i += 1
        if i > 3:
            break  # otherwise the generator would loop indefinitely


